// transcrb-capture — record a CoreAudio input device to 16 kHz mono WAV.
//
// This exists because ffmpeg's avfoundation input cannot be made reliable. It
// holds exactly one pending audio buffer and blocks the capture callback until
// the main loop reads it (libavdevice/avfoundation.m), so the whole capture has
// about one buffer period — roughly 10ms — of tolerance. CoreAudio runs in real
// time and cannot wait, so anything produced while that callback is blocked is
// discarded before ffmpeg ever sees it. On a busy machine that costs 13% of a
// meeting; under the macOS background throttle it cost 80%, and two of Adin's
// recorded meetings with it.
//
// The design here inverts that. The capture callback does nothing but downmix
// to mono and memcpy into a ring buffer holding thirty seconds of audio; a
// separate thread resamples and writes to disk. Nothing on the audio side ever
// waits for the disk, and the writer has to fall thirty seconds behind — not
// ten milliseconds — before a single sample is lost.
//
// It also *counts* what it loses rather than leaving it to be inferred. Gaps in
// the device's own sample clock are detected, filled with silence so every later
// timestamp stays honest, and reported on stdout at exit.
//
//   transcrb-capture --device-uid <uid> --out <path.wav> [--rate 16000]
//                    [--ring-seconds 30] [--seconds N]
//
// Stops on 'q' or EOF on stdin (matching ffmpeg, so the caller is unchanged), on
// SIGINT/SIGTERM, or after --seconds. Prints one JSON object on stdout at exit.

import AVFoundation
import Foundation

// ---- options -----------------------------------------------------------------

struct Options {
    var deviceUID = ""
    var deviceIndex = -1
    var output = ""
    var rate: Double = 16000
    var ringSeconds: Double = 30
    var seconds: Double = 0          // 0 = until told to stop
    var progress = ""                // where to publish how it is going
    var listDevices = false
}

func fail(_ message: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data("transcrb-capture: \(message)\n".utf8))
    exit(code)
}

func parseOptions() -> Options {
    var o = Options()
    var args = CommandLine.arguments.dropFirst().makeIterator()
    while let a = args.next() {
        switch a {
        case "--device-uid":   o.deviceUID = args.next() ?? ""
        case "--device-index": o.deviceIndex = Int(args.next() ?? "") ?? -1
        case "--out":          o.output = args.next() ?? ""
        case "--rate":         o.rate = Double(args.next() ?? "") ?? 16000
        case "--ring-seconds": o.ringSeconds = Double(args.next() ?? "") ?? 30
        case "--seconds":      o.seconds = Double(args.next() ?? "") ?? 0
        case "--progress":     o.progress = args.next() ?? ""
        case "--list-devices": o.listDevices = true
        default: fail("unknown argument \(a)", code: 2)
        }
    }
    return o
}

// ---- CoreAudio device lookup -------------------------------------------------

func systemDeviceIDs() -> [AudioDeviceID] {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(
        AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size) == noErr,
        size > 0 else { return [] }
    var ids = [AudioDeviceID](repeating: 0,
                              count: Int(size) / MemoryLayout<AudioDeviceID>.size)
    guard AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &ids) == noErr
        else { return [] }
    return ids
}

func stringProperty(_ id: AudioDeviceID, _ selector: AudioObjectPropertySelector) -> String? {
    var addr = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var size = UInt32(MemoryLayout<CFString?>.size)
    var value: CFString? = nil
    let status = withUnsafeMutablePointer(to: &value) {
        AudioObjectGetPropertyData(id, &addr, 0, nil, &size, $0)
    }
    guard status == noErr, let value else { return nil }
    return value as String
}

func inputChannelCount(_ id: AudioDeviceID) -> Int {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyStreamConfiguration,
        mScope: kAudioObjectPropertyScopeInput,
        mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(id, &addr, 0, nil, &size) == noErr,
          size > 0 else { return 0 }
    let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size),
                                               alignment: MemoryLayout<AudioBufferList>.alignment)
    defer { raw.deallocate() }
    guard AudioObjectGetPropertyData(id, &addr, 0, nil, &size, raw) == noErr else { return 0 }
    let list = UnsafeMutableAudioBufferListPointer(
        raw.assumingMemoryBound(to: AudioBufferList.self))
    return list.reduce(0) { $0 + Int($1.mNumberChannels) }
}

func deviceID(forUID uid: String) -> AudioDeviceID? {
    systemDeviceIDs().first { stringProperty($0, kAudioDevicePropertyDeviceUID) == uid }
}

/// The UID behind an ffmpeg-style ":N" input id.
///
/// A position is not an identity, and this function is the proof. It used to
/// walk `[AVCaptureDevice devicesWithMediaType:]`, the deprecated list, while
/// ffmpeg had moved to a discovery session — which sorts the same devices
/// differently. With a Continuity iPhone microphone present the two lists
/// disagreed from index 0, so the server would say "record index 4" meaning the
/// system-audio tap and the recorder would open something else. It opened, it
/// never errored, and it delivered digital silence for three hours.
///
/// The discovery session is used now so the two agree, but nothing in Transcrb
/// addresses a device this way any longer: `--device-uid` is exact, and this
/// remains only so an ffmpeg-only device listing is still usable by hand.
func uidForAVFoundationIndex(_ index: Int) -> String? {
    let session = AVCaptureDevice.DiscoverySession(
        deviceTypes: [.microphone, .external], mediaType: .audio, position: .unspecified)
    let devices = session.devices
    guard index >= 0, index < devices.count else { return nil }
    FileHandle.standardError.write(Data(
        "transcrb-capture: resolving a device by position is ambiguous; prefer --device-uid\n".utf8))
    return devices[index].uniqueID
}

// ---- WAV output --------------------------------------------------------------
//
// Written incrementally with the data-chunk size left at zero until close, so a
// recording in progress is a readable file rather than a promise: the live meter
// reads it while the meeting runs, and a crash mid-meeting leaves everything
// captured up to that moment on disk instead of nothing.

final class WavWriter {
    private let handle: FileHandle
    private var dataBytes: UInt32 = 0

    init(path: String, rate: UInt32) throws {
        FileManager.default.createFile(atPath: path, contents: nil)
        guard let h = FileHandle(forWritingAtPath: path) else {
            throw NSError(domain: "transcrb", code: 1, userInfo:
                [NSLocalizedDescriptionKey: "cannot open \(path) for writing"])
        }
        handle = h
        var header = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { header.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { header.append(contentsOf: $0) } }
        header.append(contentsOf: Array("RIFF".utf8)); u32(0)
        header.append(contentsOf: Array("WAVE".utf8))
        header.append(contentsOf: Array("fmt ".utf8)); u32(16)
        u16(1)                      // PCM
        u16(1)                      // mono
        u32(rate)
        u32(rate * 2)               // byte rate
        u16(2)                      // block align
        u16(16)                     // bits
        header.append(contentsOf: Array("data".utf8)); u32(0)
        handle.write(header)
    }

    func write(_ samples: UnsafeBufferPointer<Int16>) {
        guard let base = samples.baseAddress, !samples.isEmpty else { return }
        handle.write(Data(bytes: base, count: samples.count * 2))
        dataBytes &+= UInt32(samples.count * 2)
    }

    /// Patch the two sizes so the file is a well-formed wav for anything that
    /// trusts the header, then close.
    func close() {
        func patch(_ offset: UInt64, _ value: UInt32) {
            try? handle.seek(toOffset: offset)
            handle.write(withUnsafeBytes(of: value.littleEndian) { Data($0) })
        }
        patch(4, 36 &+ dataBytes)
        patch(40, dataBytes)
        try? handle.close()
    }
}

// ---- ring buffer -------------------------------------------------------------
//
// One producer (the capture callback) and one consumer (the writer thread). The
// producer holds the lock only for a memcpy of a few thousand floats and never
// blocks on the disk; the consumer copies out under the lock and writes outside
// it. Thirty seconds of headroom at the device's own rate.

final class MonoRing {
    private var storage: [Float]
    private var readIndex = 0
    private var writeIndex = 0
    private var available = 0
    private let condition = NSCondition()
    private(set) var overflowFrames = 0
    private var closed = false

    init(capacity: Int) {
        storage = [Float](repeating: 0, count: max(capacity, 1024))
    }

    /// Append frames, dropping them if the writer has fallen a whole ring behind.
    func push(_ source: UnsafePointer<Float>, count: Int) {
        condition.lock()
        defer { condition.unlock() }
        let capacity = storage.count
        if count > capacity - available {
            overflowFrames += count
            condition.signal()
            return
        }
        storage.withUnsafeMutableBufferPointer { dst in
            guard let base = dst.baseAddress else { return }
            let firstRun = min(count, capacity - writeIndex)
            base.advanced(by: writeIndex).update(from: source, count: firstRun)
            if firstRun < count {
                base.update(from: source.advanced(by: firstRun), count: count - firstRun)
            }
        }
        writeIndex = (writeIndex + count) % capacity
        available += count
        condition.signal()
    }

    /// Append `count` frames of silence — a gap the device itself reported.
    func pushSilence(count: Int) {
        guard count > 0 else { return }
        let chunk = [Float](repeating: 0, count: min(count, 16384))
        var remaining = count
        chunk.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            while remaining > 0 {
                let n = min(remaining, buf.count)
                push(base, count: n)
                remaining -= n
            }
        }
    }

    /// Block until there is something to write, or the recording has ended.
    /// Returns an empty array only when closed and drained.
    func drain(max limit: Int) -> [Float] {
        condition.lock()
        defer { condition.unlock() }
        while available == 0 && !closed { condition.wait() }
        let count = min(available, limit)
        guard count > 0 else { return [] }
        var out = [Float](repeating: 0, count: count)
        let capacity = storage.count
        storage.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            out.withUnsafeMutableBufferPointer { dst in
                guard let d = dst.baseAddress else { return }
                let firstRun = min(count, capacity - readIndex)
                d.update(from: base.advanced(by: readIndex), count: firstRun)
                if firstRun < count {
                    d.advanced(by: firstRun).update(from: base, count: count - firstRun)
                }
            }
        }
        readIndex = (readIndex + count) % capacity
        available -= count
        return out
    }

    func close() {
        condition.lock()
        closed = true
        condition.broadcast()
        condition.unlock()
    }

    var isFinished: Bool {
        condition.lock()
        defer { condition.unlock() }
        return closed && available == 0
    }

    var isEmpty: Bool {
        condition.lock()
        defer { condition.unlock() }
        return available == 0
    }
}

// ---- stop signalling ---------------------------------------------------------

final class StopFlag {
    private let lock = NSCondition()
    private var stopped = false
    private(set) var reason = ""

    func stop(_ why: String) {
        lock.lock()
        if !stopped { stopped = true; reason = why }
        lock.broadcast()
        lock.unlock()
    }

    var isStopped: Bool {
        lock.lock()
        defer { lock.unlock() }
        return stopped
    }

    /// Wait for a stop, or until `deadline` if one was given.
    func wait(seconds: Double) {
        lock.lock()
        let end = seconds > 0 ? Date().addingTimeInterval(seconds) : Date.distantFuture
        while !stopped && Date() < end { lock.wait(until: min(end, Date().addingTimeInterval(0.25))) }
        if !stopped { stopped = true; reason = "duration" }
        lock.unlock()
    }
}

let stopFlag = StopFlag()

// ---- main --------------------------------------------------------------------

let options = parseOptions()

if options.listDevices {
    var out: [String] = []
    for id in systemDeviceIDs() where inputChannelCount(id) > 0 {
        let uid = stringProperty(id, kAudioDevicePropertyDeviceUID) ?? ""
        let name = stringProperty(id, kAudioObjectPropertyName) ?? ""
        let escape = { (s: String) in s.replacingOccurrences(of: "\"", with: "\\\"") }
        out.append("""
        {"uid":"\(escape(uid))","name":"\(escape(name))","inputs":\(inputChannelCount(id))}
        """)
    }
    print("[" + out.joined(separator: ",") + "]")
    exit(0)
}

guard !options.output.isEmpty else { fail("--out is required", code: 2) }

var wantedUID = options.deviceUID
if wantedUID.isEmpty, options.deviceIndex >= 0 {
    guard let resolved = uidForAVFoundationIndex(options.deviceIndex) else {
        fail("no audio input at index \(options.deviceIndex)")
    }
    wantedUID = resolved
}
guard !wantedUID.isEmpty else { fail("--device-uid or --device-index is required", code: 2) }
guard let device = deviceID(forUID: wantedUID) else {
    fail("no input device with UID \(wantedUID)")
}

// Written into every progress and summary line. Which device a recording was
// actually taken from is the first question asked of a recording that came back
// silent, and it used to be unanswerable after the fact.
let openedUID = wantedUID
let openedName = stringProperty(device, kAudioObjectPropertyName) ?? ""

let engine = AVAudioEngine()
let input = engine.inputNode

// AVAudioEngine binds its input to the default device on first touch; this
// repoints it at the device we were asked for. The format has to be re-read
// afterwards, because it belongs to the new device, not the old one.
var deviceRef = device
guard let unit = input.audioUnit else { fail("the input node has no audio unit") }
let setStatus = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice,
                                     kAudioUnitScope_Global, 0, &deviceRef,
                                     UInt32(MemoryLayout<AudioDeviceID>.size))
guard setStatus == noErr else {
    fail("could not select the capture device (CoreAudio status \(setStatus))")
}

let inputFormat = input.inputFormat(forBus: 0)
guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0 else {
    fail("the capture device reports no input channels")
}

guard let outFormat = AVAudioFormat(commonFormat: .pcmFormatInt16,
                                    sampleRate: options.rate,
                                    channels: 1, interleaved: true) else {
    fail("could not build the 16 kHz mono output format")
}

/// The parts of the chain that belong to the input device's current format.
///
/// They are replaceable because the device's format can change underneath a
/// running recording. A Bluetooth headset joining a call switches from the
/// 48 kHz playback profile to the 16 kHz headset one, and that is a meeting, not
/// an edge case. AVAudioEngine stops when it happens.
final class Chain: @unchecked Sendable {
    let lock = NSLock()
    var input: AVAudioFormat
    var mono: AVAudioFormat
    var converter: AVAudioConverter
    init(input: AVAudioFormat, mono: AVAudioFormat, converter: AVAudioConverter) {
        self.input = input; self.mono = mono; self.converter = converter
    }
}

func buildChain(for format: AVAudioFormat) -> (AVAudioFormat, AVAudioConverter)? {
    guard let mono = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                   sampleRate: format.sampleRate,
                                   channels: 1, interleaved: false),
          let converter = AVAudioConverter(from: mono, to: outFormat) else { return nil }
    return (mono, converter)
}

guard let (firstMono, firstConverter) = buildChain(for: inputFormat) else {
    fail("could not build the 16 kHz mono conversion")
}
let chain = Chain(input: inputFormat, mono: firstMono, converter: firstConverter)

let writer: WavWriter
do { writer = try WavWriter(path: options.output, rate: UInt32(options.rate)) }
catch { fail("\(error.localizedDescription)") }

// Sized for a rate higher than the device is running at, so that a device which
// changes its rate mid-recording does not need the ring rebuilt underneath a
// live meeting. Thirty seconds at 48 kHz is a minute at 24, which is a bonus
// rather than a problem.
let ring = MonoRing(capacity: Int(max(inputFormat.sampleRate, 96000) * options.ringSeconds))

// Everything the tap touches. The tap is the only writer and runs serially, so
// the scratch buffer is allocated once here rather than per callback: a malloc
// on the audio path can block on a lock held by any other thread on the machine,
// which is the class of stall this whole program exists to avoid.
//
// Counted in seconds rather than frames because the device's sample rate can
// change mid-recording, and adding frame counts across two rates gives a number
// that means nothing.
final class TapState: @unchecked Sendable {
    var capturedSeconds = 0.0
    var gapSeconds = 0.0
    var expectedSampleTime: AVAudioFramePosition = -1
    let scratch = UnsafeMutablePointer<Float>.allocate(capacity: 1 << 16)
    let scratchCapacity = 1 << 16
    // A device that stops delivering entirely produces no gap and no overflow:
    // the counters simply stop moving, which reads as a healthy recording that
    // happens to be short. Written by the tap, read by the watchdog. A Double
    // store is a single aligned word, so no lock is needed on the audio path.
    var lastBufferAt = Date().timeIntervalSince1970
    var stallSeconds = 0.0
    var reconfigurations = 0
}
let counters = TapState()

// Longer than any normal buffer period and shorter than anyone would want to
// go on not knowing. A device that has gone quiet for two seconds has stopped.
let STALL_GRACE_SECONDS = 2.0

// The tap: downmix and hand off. No file I/O, no allocation beyond one scratch
// buffer per call, and no path that can wait on anything slower than a memcpy.
func installCaptureTap(format: AVAudioFormat) {
    let rate = format.sampleRate
    input.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, when in
        let frames = Int(buffer.frameLength)
        guard frames > 0, let channels = buffer.floatChannelData else { return }
        let channelCount = Int(buffer.format.channelCount)

        // A jump in the device's own sample clock is audio CoreAudio produced
        // and nobody collected. Fill it so the recording stays aligned with the
        // meeting instead of every later word sliding earlier.
        if counters.expectedSampleTime >= 0 {
            let gap = when.sampleTime - counters.expectedSampleTime
            if gap > 0 {
                counters.gapSeconds += Double(gap) / rate
                ring.pushSilence(count: Int(min(gap, AVAudioFramePosition(rate * 10))))
            }
        }
        counters.expectedSampleTime = when.sampleTime + AVAudioFramePosition(frames)
        counters.lastBufferAt = Date().timeIntervalSince1970

        // Every channel of the device carries part of the meeting: the tap's
        // stereo system audio is the far side of the call and the mic channel is
        // this side, so they are averaged rather than one being chosen. Same
        // downmix ffmpeg's `-ac 1` was doing.
        var offset = 0
        while offset < frames {
            let n = min(frames - offset, counters.scratchCapacity)
            let d = counters.scratch
            d.update(repeating: 0, count: n)
            for c in 0..<channelCount {
                let src = channels[c].advanced(by: offset)
                for i in 0..<n { d[i] += src[i] }
            }
            if channelCount > 1 {
                let scale = 1.0 / Float(channelCount)
                for i in 0..<n { d[i] *= scale }
            }
            ring.push(d, count: n)
            offset += n
        }
        counters.capturedSeconds += Double(frames) / rate
    }
}
installCaptureTap(format: inputFormat)

// A recording that is losing audio has to be visible while the meeting is still
// running. Waiting until stop is how an hour of a meeting gets discovered
// missing an hour too late.
func summaryJSON(written: Int, failures: Int, reason: String) -> String {
    chain.lock.lock()
    let rate = chain.input.sampleRate
    let channels = chain.input.channelCount
    chain.lock.unlock()
    let capturedSeconds = counters.capturedSeconds
    let gapSeconds = counters.gapSeconds
    let overflowSeconds = Double(ring.overflowFrames) / rate
    let stallSeconds = counters.stallSeconds
    let stalledNow = Date().timeIntervalSince1970 - counters.lastBufferAt > STALL_GRACE_SECONDS
    // Time the device stopped delivering counts against the meeting as much as
    // time it delivered nothing useful, so it belongs in both halves.
    let total = capturedSeconds + gapSeconds + stallSeconds
    let lost = gapSeconds + overflowSeconds + stallSeconds
    func f(_ v: Double) -> String { String(format: "%.3f", v) }
    return """
    {"captured_seconds":\(f(capturedSeconds)),"gap_seconds":\(f(gapSeconds)),\
    "overflow_seconds":\(f(overflowSeconds)),"stall_seconds":\(f(stallSeconds)),\
    "stalled":\(stalledNow),"reconfigurations":\(counters.reconfigurations),\
    "dropped":\(String(format: "%.5f", total > 0 ? lost / total : 0)),\
    "written_seconds":\(f(Double(written) / options.rate)),\
    "convert_failures":\(failures),"device_rate":\(rate),\
    "device_channels":\(channels),\
    "device_uid":"\(openedUID.replacingOccurrences(of: "\"", with: "\\\""))",\
    "device_name":"\(openedName.replacingOccurrences(of: "\"", with: "\\\""))",\
    "reason":"\(reason)"}
    """
}

func publishProgress(_ json: String) {
    guard !options.progress.isEmpty else { return }
    let temp = options.progress + ".tmp"
    guard (try? Data(json.utf8).write(to: URL(fileURLWithPath: temp))) != nil else { return }
    // Rename rather than truncate-and-write: the reader is polling, and a
    // half-written file would read as a recording in trouble.
    _ = try? FileManager.default.replaceItemAt(URL(fileURLWithPath: options.progress),
                                               withItemAt: URL(fileURLWithPath: temp))
}

// The writer thread: resample and write, as far behind the meeting as it likes.
var written = 0
var convertFailures = 0
let writerThread = Thread {
    while true {
        // Drained outside the chain lock on purpose: this call blocks until
        // there is audio, and holding the lock across it would deadlock a
        // reconfiguration that is waiting to take it.
        let block = ring.drain(max: 65536)
        if block.isEmpty {
            if ring.isFinished { break }
            continue
        }
        chain.lock.lock()
        let mono = chain.mono
        let converter = chain.converter
        let rate = chain.input.sampleRate
        defer { chain.lock.unlock() }
        guard let inBuffer = AVAudioPCMBuffer(pcmFormat: mono,
                                              frameCapacity: AVAudioFrameCount(block.count))
        else { continue }
        inBuffer.frameLength = AVAudioFrameCount(block.count)
        block.withUnsafeBufferPointer { src in
            if let base = src.baseAddress, let dst = inBuffer.floatChannelData {
                dst[0].update(from: base, count: block.count)
            }
        }
        let capacity = AVAudioFrameCount(Double(block.count) * (options.rate / rate)) + 1024
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat,
                                               frameCapacity: capacity) else { continue }
        var supplied = false
        var error: NSError?
        converter.convert(to: outBuffer, error: &error) { _, status in
            if supplied { status.pointee = .noDataNow; return nil }
            supplied = true
            status.pointee = .haveData
            return inBuffer
        }
        // Counted rather than printed. A per-failure line on stderr would fill
        // the pipe the caller is not reading and block this thread, which is the
        // exact stall this program exists to avoid; the count rides out in the
        // summary instead.
        if error != nil { convertFailures += 1; continue }
        if let samples = outBuffer.int16ChannelData, outBuffer.frameLength > 0 {
            let count = Int(outBuffer.frameLength)
            writer.write(UnsafeBufferPointer(start: samples[0], count: count))
            written += count
        }
    }
}
writerThread.start()

// A device can change format underneath a running recording: a Bluetooth headset
// joining a call drops from the 48 kHz playback profile to the 16 kHz headset
// one, and AVAudioEngine stops when that happens. Left alone the recording would
// simply end there, with the file looking healthy up to the moment it went
// quiet. This restarts the capture around the new format and carries on.
//
// Nothing here runs unless the notification fires, so the ordinary path is
// untouched by it.
// Delivered on a queue of its own rather than on whichever thread posted it:
// the handler stops and restarts the engine and waits on the writer, and doing
// that on an AVAudioEngine internal thread invites a deadlock.
let reconfigureQueue = OperationQueue()
reconfigureQueue.maxConcurrentOperationCount = 1

/// Rebuild the capture chain around whatever the device looks like now.
///
/// `force` skips the "did anything actually change" test, for the case where
/// nothing was posted at all and the only evidence is that audio stopped
/// arriving. Serialised onto `reconfigureQueue`, so the notification and the
/// watchdog can never rebuild at the same time.
func rebuildCapture(force: Bool) {
    guard !stopFlag.isStopped else { return }

    // AVAudioEngine posts a notification shortly after the input device is set,
    // when nothing has actually changed. Restarting on it would cost 75ms of
    // every recording for nothing. A real change either alters the format or
    // stops the engine.
    let current = input.inputFormat(forBus: 0)
    chain.lock.lock()
    let unchanged = current.sampleRate == chain.input.sampleRate
        && current.channelCount == chain.input.channelCount
    chain.lock.unlock()
    if !force && unchanged && engine.isRunning { return }

    let downAt = Date().timeIntervalSince1970
    input.removeTap(onBus: 0)
    engine.stop()
    // Let the writer catch up before the format changes underneath it, so no
    // block is ever converted at the wrong rate. Bounded: a writer that is not
    // going to catch up must not hold the recording down with it.
    let waitUntil = Date().timeIntervalSince1970 + 2.0
    while !ring.isEmpty && Date().timeIntervalSince1970 < waitUntil { usleep(20_000) }

    guard let again = deviceID(forUID: wantedUID) else {
        FileHandle.standardError.write(Data(
            "transcrb-capture: the capture device is gone; recording stops here\n".utf8))
        stopFlag.stop("device-lost")
        return
    }
    var ref = again
    if let unit = input.audioUnit {
        _ = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice,
                                 kAudioUnitScope_Global, 0, &ref,
                                 UInt32(MemoryLayout<AudioDeviceID>.size))
    }
    let format = input.inputFormat(forBus: 0)
    guard format.sampleRate > 0, format.channelCount > 0,
          let (mono, converter) = buildChain(for: format) else {
        FileHandle.standardError.write(Data(
            "transcrb-capture: the capture device came back unusable\n".utf8))
        stopFlag.stop("device-unusable")
        return
    }
    chain.lock.lock()
    chain.input = format; chain.mono = mono; chain.converter = converter
    chain.lock.unlock()

    // The clock belongs to a different stream now, so the next buffer's sample
    // time says nothing about the last one's.
    counters.expectedSampleTime = -1
    installCaptureTap(format: format)
    do { try engine.start() } catch {
        FileHandle.standardError.write(Data(
            "transcrb-capture: could not restart capture: \(error)\n".utf8))
        stopFlag.stop("restart-failed")
        return
    }
    // The seconds the device was down are missing from the meeting whether or
    // not the restart worked, so they are written in as silence and counted.
    let lost = Date().timeIntervalSince1970 - downAt
    if lost > 0 {
        counters.gapSeconds += lost
        ring.pushSilence(count: Int(min(lost, 30) * format.sampleRate))
    }
    counters.lastBufferAt = Date().timeIntervalSince1970
    counters.reconfigurations += 1
}

let reconfigureObserver = NotificationCenter.default.addObserver(
    forName: .AVAudioEngineConfigurationChange, object: engine,
    queue: reconfigureQueue) { _ in rebuildCapture(force: false) }

// Progress is published from here, not from the writer thread. The writer
// blocks waiting for audio, so a device that stops delivering would freeze the
// progress file at its last healthy reading — a recording capturing nothing
// while reporting 100% captured, which is worse than saying nothing at all.
// Started once the engine is running, so the clock it measures against is the
// moment audio should have begun arriving. A device that never delivers a single
// buffer then reads as wholly lost, which is what it is.
// How long to let a silent device stay silent before rebuilding it rather than
// waiting to be told. Measured on this machine 2026-08-20: swapping the audio
// output mid-recording stopped delivery for 31.6 seconds before macOS posted a
// configuration change at all, and every second of that is simply absent from
// the recording. Waiting for the notification is not enough — sometimes it does
// not come until the device feels like it. Comfortably longer than the largest
// legitimate pause between buffers, so an ordinary recording never rebuilds.
// 1.5s is roughly 17 times the interval between buffers (4096 frames at 48 kHz
// is 85ms), so ordinary jitter cannot reach it, while every second above it is
// meeting that never lands in the file. Measured with headphones swapped
// mid-recording: at 3.0s the hole was 3.5s and four spoken numbers went missing.
let RECOVER_AFTER_SECONDS = 1.5
let RECOVER_RETRY_SECONDS = 5.0

let watchdogThread = Thread {
    var lastTick = Date().timeIntervalSince1970
    var lastRecoveryAt = 0.0
    while !stopFlag.isStopped {
        Thread.sleep(forTimeInterval: 0.5)
        let now = Date().timeIntervalSince1970
        let quietFor = now - counters.lastBufferAt
        if quietFor > STALL_GRACE_SECONDS {
            counters.stallSeconds += now - lastTick
        }
        // Nothing is arriving and nothing told us why. Rebuild around whatever
        // the device is now, then leave it a moment before trying again, so a
        // genuinely dead input is not restarted twice a second for an hour.
        if quietFor > RECOVER_AFTER_SECONDS,
           now - lastRecoveryAt > RECOVER_RETRY_SECONDS,
           reconfigureQueue.operationCount == 0 {
            lastRecoveryAt = now
            let why = "transcrb-capture: no audio for "
                + String(format: "%.1f", quietFor)
                + "s; rebuilding the capture rather than waiting to be told\n"
            FileHandle.standardError.write(Data(why.utf8))
            reconfigureQueue.addOperation { rebuildCapture(force: true) }
        }
        lastTick = now
        publishProgress(summaryJSON(written: written, failures: convertFailures,
                                    reason: "recording"))
    }
}

// Stop on 'q' or EOF on stdin, the way ffmpeg does, so the caller is unchanged.
// Skipped for a fixed-length capture: the level check runs with stdin closed,
// and an instant EOF would end a 1.5-second probe before it recorded anything.
if options.seconds <= 0 {
    let stdinThread = Thread {
        while true {
            let data = FileHandle.standardInput.availableData
            if data.isEmpty { stopFlag.stop("stdin-eof"); return }
            if data.contains(UInt8(ascii: "q")) { stopFlag.stop("stdin-q"); return }
        }
    }
    stdinThread.start()
}

// Held for the process lifetime on purpose: a released source stops firing, and
// a SIGTERM that does nothing would strand the recording instead of closing it.
var signalSources: [DispatchSourceSignal] = []
for sig in [SIGINT, SIGTERM] {
    signal(sig, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: sig, queue: .global())
    source.setEventHandler { stopFlag.stop("signal") }
    source.resume()
    signalSources.append(source)
}

counters.lastBufferAt = Date().timeIntervalSince1970
do { try engine.start() }
catch { fail("could not start capture: \(error.localizedDescription)") }
watchdogThread.start()

stopFlag.wait(seconds: options.seconds)

// Shutting down must not be able to hang. `engine.stop()` blocks indefinitely
// when the device it is bound to has gone away, and the writer join below span
// on `isFinished` with no bound, so a recorder asked to stop could keep the tap
// and its file open forever: on 2026-08-20 one ignored SIGTERM for eight
// minutes and had to be killed. The samples are already on disk — the WAV is
// written as it goes — so leaving by the back door costs nothing but a header,
// and the header is repaired on the way in.
let shutdownGuard = Thread {
    Thread.sleep(forTimeInterval: 10.0)
    FileHandle.standardError.write(Data(
        "transcrb-capture: shutdown did not complete in 10s; exiting anyway\n".utf8))
    publishProgress(summaryJSON(written: written, failures: convertFailures,
                                reason: stopFlag.reason + "-forced"))
    exit(0)
}
shutdownGuard.start()

NotificationCenter.default.removeObserver(reconfigureObserver)
engine.stop()
input.removeTap(onBus: 0)
ring.close()
let joinUntil = Date().addingTimeInterval(5.0)
while !writerThread.isFinished && Date() < joinUntil { usleep(20_000) }
writer.close()

let summary = summaryJSON(written: written, failures: convertFailures,
                          reason: stopFlag.reason)
publishProgress(summary)
print(summary)
exit(0)
