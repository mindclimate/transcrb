"""Create the two CoreAudio devices that make a call recordable on macOS.

Installing BlackHole only adds a virtual output. On its own it records silence,
because it carries audio solely while macOS is playing into it, and it never
carries your own voice. Two aggregate devices close that gap:

  * a Multi-Output Device (headphones + BlackHole) so the call is audible *and*
    copied to BlackHole
  * an Aggregate Device (microphone + BlackHole) so a recording contains both
    the remote side and you

Audio MIDI Setup builds these by hand; CoreAudio's AudioHardwareCreateAggregateDevice
builds them from here so a fresh install arrives ready to record. Bound with
ctypes rather than pyobjc to avoid adding a dependency for one call.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import plistlib
from dataclasses import dataclass

OUTPUT_DEVICE_NAME = "Transcrb Output (hear + capture)"
OUTPUT_DEVICE_UID = "com.transcrb.multioutput"
INPUT_DEVICE_NAME = "Transcrb Input (call + mic)"
INPUT_DEVICE_UID = "com.transcrb.aggregate.input"

# The tap-based device, built on demand while recording.
CAPTURE_DEVICE_NAME = "Transcrb Capture"
CAPTURE_DEVICE_UID = "com.transcrb.capture"
TAP_NAME = "Transcrb System Audio"

OUR_UIDS = (OUTPUT_DEVICE_UID, INPUT_DEVICE_UID, CAPTURE_DEVICE_UID)

# macOS 14.2 is where AudioHardwareCreateProcessTap arrived.
MIN_TAP_VERSION = (14, 2)

@dataclass(frozen=True)
class AudioDevice:
    uid: str
    name: str
    inputs: int
    outputs: int
    id: int = 0

    @property
    def is_blackhole(self) -> bool:
        return "blackhole" in self.name.lower()

# ---- planning (pure) ---------------------------------------------------------
# The CoreAudio key names below are the string constants from
# <CoreAudio/AudioHardwareBase.h>: kAudioAggregateDeviceNameKey and friends.

def aggregate_description(name: str, uid: str, members: list[tuple[str, bool]], *,
                          stacked: bool, master_uid: str) -> dict:
    """A description dictionary for AudioHardwareCreateAggregateDevice.

    `members` is (device uid, needs drift compensation). `stacked` selects a
    Multi-Output Device over a plain Aggregate Device.
    """
    return {
        "name": name,
        "uid": uid,
        "master": master_uid,
        "private": 0,       # 1 would make the device vanish when we exit
        "stacked": 1 if stacked else 0,
        "subdevices": [{"uid": m_uid, "drift": 1 if drift else 0}
                       for m_uid, drift in members],
    }

def find_blackhole(devices: list[AudioDevice]) -> AudioDevice | None:
    return next((d for d in devices if d.is_blackhole), None)

def _pick(devices: list[AudioDevice], preferred_uid: str,
          channels: str) -> AudioDevice | None:
    """The device to pair with BlackHole: the current default where sensible.

    Skips BlackHole itself, the devices this module creates (nesting one inside
    itself is invalid), and anything without channels in the right direction.
    """
    usable = [d for d in devices
              if getattr(d, channels) > 0
              and not d.is_blackhole
              and d.uid not in OUR_UIDS]
    if not usable:
        return None
    # A Continuity iPhone leaves with the phone and would break the device
    # mid-meeting, so it loses even to a non-default built-in mic.
    pool = [d for d in usable if not _is_continuity(d)] or usable
    preferred = next((d for d in pool if d.uid == preferred_uid), None)
    if preferred is not None:
        return preferred
    return next((d for d in pool if "built" in d.uid.lower()), pool[0])

def _is_continuity(device: AudioDevice) -> bool:
    return any(k in device.name.lower() for k in ("iphone", "ipad"))

def output_plan(devices: list[AudioDevice], default_output_uid: str) -> dict | None:
    """Multi-Output: the real output stays the clock master, BlackHole drifts."""
    blackhole = find_blackhole(devices)
    speakers = _pick(devices, default_output_uid, "outputs")
    if blackhole is None or speakers is None:
        return None
    return aggregate_description(
        OUTPUT_DEVICE_NAME, OUTPUT_DEVICE_UID,
        [(speakers.uid, False), (blackhole.uid, True)],
        stacked=True, master_uid=speakers.uid)

def input_plan(devices: list[AudioDevice], default_input_uid: str) -> dict | None:
    """Aggregate: the microphone clocks it, BlackHole drifts."""
    blackhole = find_blackhole(devices)
    mic = _pick(devices, default_input_uid, "inputs")
    if blackhole is None or mic is None:
        return None
    return aggregate_description(
        INPUT_DEVICE_NAME, INPUT_DEVICE_UID,
        [(mic.uid, False), (blackhole.uid, True)],
        stacked=False, master_uid=mic.uid)

def capture_description(tap_uid: str, mic_uid: str | None) -> dict:
    """An aggregate of a system-audio tap plus the microphone.

    This is the device that needs no configuration: the tap follows whatever
    macOS is playing, so the active output device is irrelevant and swapping
    headphones mid-call changes nothing. The mic rides along so the recording
    has both sides of the conversation.
    """
    desc = {
        "name": CAPTURE_DEVICE_NAME,
        "uid": CAPTURE_DEVICE_UID,
        "private": 0,      # ffmpeg is a separate process and must be able to open it
        "stacked": 0,
        "subdevices": [{"uid": mic_uid, "drift": 0}] if mic_uid else [],
        # The tap has no clock of its own, so it drifts against the mic.
        "taps": [{"uid": tap_uid, "drift": 1}],
    }
    desc["master"] = mic_uid or ""
    return desc

def taps_supported(version: tuple[int, ...] | None = None) -> bool:
    if version is None:
        version = macos_version()
    return tuple(version[:2]) >= MIN_TAP_VERSION

def macos_version() -> tuple[int, ...]:
    import platform
    parts = platform.mac_ver()[0].split(".")
    return tuple(int(p) for p in parts if p.isdigit()) or (0,)

# ---- CoreAudio bindings ------------------------------------------------------

kAudioObjectSystemObject = 1

def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")

class _PropertyAddress(ctypes.Structure):
    _fields_ = [("mSelector", ctypes.c_uint32),
                ("mScope", ctypes.c_uint32),
                ("mElement", ctypes.c_uint32)]

class _AudioBuffer(ctypes.Structure):
    _fields_ = [("mNumberChannels", ctypes.c_uint32),
                ("mDataByteSize", ctypes.c_uint32),
                ("mData", ctypes.c_void_p)]

def _frameworks():
    ca = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
    cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
    cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                      ctypes.c_long, ctypes.c_uint32]
    cf.CFDataCreate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long]
    cf.CFDataCreate.restype = ctypes.c_void_p
    cf.CFPropertyListCreateWithData.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                ctypes.c_uint32, ctypes.c_void_p,
                                                ctypes.c_void_p]
    cf.CFPropertyListCreateWithData.restype = ctypes.c_void_p
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    return ca, cf

def _address(selector: str, scope: str = "glob") -> _PropertyAddress:
    return _PropertyAddress(_fourcc(selector), _fourcc(scope), 0)

def _get_property(ca, obj: int, addr: _PropertyAddress, ctype):
    size = ctypes.c_uint32(0)
    if ca.AudioObjectGetPropertyDataSize(obj, ctypes.byref(addr), 0, None,
                                         ctypes.byref(size)) != 0:
        return None
    if size.value == 0:
        return None
    buf = ctypes.create_string_buffer(size.value)
    if ca.AudioObjectGetPropertyData(obj, ctypes.byref(addr), 0, None,
                                     ctypes.byref(size), buf) != 0:
        return None
    return ctypes.cast(buf, ctypes.POINTER(ctype)), size.value, buf

def _cfstring(cf, ref: int) -> str:
    out = ctypes.create_string_buffer(512)
    if not ref or not cf.CFStringGetCString(ref, out, 512, 0x08000100):  # UTF-8
        return ""
    return out.value.decode("utf-8", "replace")

def _channel_count(ca, device_id: int, scope: str) -> int:
    got = _get_property(ca, device_id, _address("slay", scope), ctypes.c_uint32)
    if got is None:
        return 0
    ptr, size, buf = got
    count = ptr[0]
    channels = 0
    offset = ctypes.sizeof(ctypes.c_uint32)
    # An AudioBufferList is a count followed by that many AudioBuffers; the
    # struct is padded to pointer alignment on arm64.
    offset = max(offset, 8)
    for _ in range(count):
        if offset + ctypes.sizeof(_AudioBuffer) > size:
            break
        chunk = bytes(buf[offset:offset + ctypes.sizeof(_AudioBuffer)])
        buffer = _AudioBuffer.from_buffer_copy(chunk)
        channels += buffer.mNumberChannels
        offset += ctypes.sizeof(_AudioBuffer)
    return channels

def list_devices() -> list[AudioDevice]:
    ca, cf = _frameworks()
    got = _get_property(ca, kAudioObjectSystemObject, _address("dev#"), ctypes.c_uint32)
    if got is None:
        return []
    ptr, size, _ = got
    devices = []
    for i in range(size // ctypes.sizeof(ctypes.c_uint32)):
        dev_id = ptr[i]
        uid_got = _get_property(ca, dev_id, _address("uid "), ctypes.c_void_p)
        name_got = _get_property(ca, dev_id, _address("lnam"), ctypes.c_void_p)
        if uid_got is None or name_got is None:
            continue
        devices.append(AudioDevice(
            uid=_cfstring(cf, uid_got[0][0]),
            name=_cfstring(cf, name_got[0][0]),
            inputs=_channel_count(ca, dev_id, "inpt"),
            outputs=_channel_count(ca, dev_id, "outp"),
            id=dev_id))
    return devices

def default_device_uid(devices: list[AudioDevice], which: str) -> str:
    """UID of the current default input ('dIn ') or output ('dOut') device."""
    ca, _ = _frameworks()
    got = _get_property(ca, kAudioObjectSystemObject, _address(which), ctypes.c_uint32)
    if got is None:
        return ""
    dev_id = got[0][0]
    return next((d.uid for d in devices if d.id == dev_id), "")

def create_aggregate_device(description: dict) -> int:
    """Create the device and return its CoreAudio id. Raises on failure."""
    ca, cf = _frameworks()
    xml = plistlib.dumps(description)
    data = cf.CFDataCreate(None, xml, len(xml))
    plist = cf.CFPropertyListCreateWithData(None, data, 0, None, None)
    if not plist:
        cf.CFRelease(data)
        raise RuntimeError("could not build the device description")
    device_id = ctypes.c_uint32(0)
    status = ca.AudioHardwareCreateAggregateDevice(
        ctypes.c_void_p(plist), ctypes.byref(device_id))
    cf.CFRelease(plist)
    cf.CFRelease(data)
    if status != 0 or device_id.value == 0:
        raise RuntimeError(f"CoreAudio refused the device (status {status})")
    return device_id.value

def destroy_aggregate_device(device_id: int) -> None:
    ca, _ = _frameworks()
    ca.AudioHardwareDestroyAggregateDevice(ctypes.c_uint32(device_id))

# ---- system-audio tap --------------------------------------------------------
# CATapDescription is an Objective-C class, so creating a tap means a handful of
# objc_msgSend calls. Done with ctypes to keep pyobjc out of the dependencies.

_UTF8 = 0x08000100

def _objc():
    objc = ctypes.CDLL(ctypes.util.find_library("objc"))
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    return objc

def _send(objc, receiver, selector: bytes, *args, restype: object = ctypes.c_void_p,
          argtypes=()):
    fn = objc.objc_msgSend
    fn.restype = restype
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
    return fn(ctypes.c_void_p(receiver), objc.sel_registerName(selector), *args)

def create_system_tap(name: str = TAP_NAME) -> tuple[int, str]:
    """Create a global system-audio tap. Returns (tap id, tap UID).

    Excludes no processes, so it hears the whole system: the call, and anything
    else playing. Left unmuted so the user still hears the meeting.
    """
    objc, cf = _objc(), _frameworks()[1]
    cf.CFArrayCreate.restype = ctypes.c_void_p
    cf.CFArrayCreate.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_long, ctypes.c_void_p]
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]

    cls = objc.objc_getClass(b"CATapDescription")
    if not cls:
        raise RuntimeError("CATapDescription unavailable (needs macOS 14.2+)")
    empty = cf.CFArrayCreate(None, None, 0, None)
    desc = _send(objc, _send(objc, cls, b"alloc"),
                 b"initStereoGlobalTapButExcludeProcesses:", empty,
                 argtypes=[ctypes.c_void_p])
    if not desc:
        raise RuntimeError("could not build a tap description")
    _send(objc, desc, b"setName:", cf.CFStringCreateWithCString(None, name.encode(), _UTF8),
          restype=None, argtypes=[ctypes.c_void_p])
    # Public, or the aggregate device would be invisible to ffmpeg.
    _send(objc, desc, b"setPrivate:", False, restype=None, argtypes=[ctypes.c_bool])
    _send(objc, desc, b"setMuteBehavior:", 0, restype=None, argtypes=[ctypes.c_long])

    ca = _frameworks()[0]
    ca.AudioHardwareCreateProcessTap.argtypes = [ctypes.c_void_p,
                                                 ctypes.POINTER(ctypes.c_uint32)]
    tap_id = ctypes.c_uint32(0)
    status = ca.AudioHardwareCreateProcessTap(ctypes.c_void_p(desc), ctypes.byref(tap_id))
    if status != 0 or tap_id.value == 0:
        raise RuntimeError(f"CoreAudio refused the system-audio tap (status {status})")
    uid = _cfstring(cf, _send(objc, _send(objc, desc, b"UUID"), b"UUIDString"))
    if not uid:
        ca.AudioHardwareDestroyProcessTap(ctypes.c_uint32(tap_id.value))
        raise RuntimeError("the tap has no UUID to reference")
    return tap_id.value, uid

def destroy_system_tap(tap_id: int) -> None:
    _frameworks()[0].AudioHardwareDestroyProcessTap(ctypes.c_uint32(tap_id))

def _device_by_uid(uid: str) -> AudioDevice | None:
    return next((d for d in list_devices() if d.uid == uid), None)

def remove_stale_capture_device() -> None:
    """Drop a capture device orphaned by a previous run that died mid-recording."""
    stale = _device_by_uid(CAPTURE_DEVICE_UID)
    if stale is not None:
        try:
            destroy_aggregate_device(stale.id)
        except Exception:
            pass

class SystemCapture:
    """A recordable input carrying system audio plus the microphone.

    Lives only as long as it is needed: the tap belongs to this process, so the
    device disappears when the recording stops or the server exits.
    """

    def __init__(self) -> None:
        self.tap_id: int | None = None
        self.device_id: int | None = None
        self.name = CAPTURE_DEVICE_NAME

    def start(self) -> str:
        """Create the device and return its avfoundation input id (":N")."""
        remove_stale_capture_device()
        self.tap_id, tap_uid = create_system_tap()
        devices = list_devices()
        mic_uid = default_device_uid(devices, "dIn ")
        mic = next((d for d in devices if d.uid == mic_uid), None)
        if mic is None or mic.inputs == 0 or mic.uid in OUR_UIDS:
            # No usable default mic: fall back to any real input, else tap only.
            mic = _pick(devices, "", "inputs")
        try:
            self.device_id = create_aggregate_device(
                capture_description(tap_uid, mic.uid if mic else None))
        except Exception:
            self.stop()
            raise
        # ffmpeg addresses avfoundation inputs by index, so the freshly created
        # device has to be located in its listing by name.
        from engine.record import list_input_devices
        entry = next((d for d in list_input_devices("darwin")
                      if d["name"].strip() == CAPTURE_DEVICE_NAME), None)
        if entry is None:
            self.stop()
            raise RuntimeError("the capture device was created but ffmpeg cannot see it")
        return entry["id"]

    def stop(self) -> None:
        if self.device_id:
            try:
                destroy_aggregate_device(self.device_id)
            except Exception:
                pass
            self.device_id = None
        if self.tap_id:
            try:
                destroy_system_tap(self.tap_id)
            except Exception:
                pass
            self.tap_id = None

# ---- setup entry point -------------------------------------------------------

def ensure_devices(log=print) -> int:
    """Create both devices if they are missing. Returns how many were created.

    Idempotent: re-running after the devices exist changes nothing, so setup can
    call it on every run.
    """
    devices = list_devices()
    if find_blackhole(devices) is None:
        log("! BlackHole 2ch not found. It needs a reboot after install — "
            "reboot, then run this again to finish the audio setup.")
        return 0

    existing = {d.uid for d in devices}
    default_out = default_device_uid(devices, "dOut")
    default_in = default_device_uid(devices, "dIn ")
    created = 0
    for uid, plan in ((OUTPUT_DEVICE_UID, output_plan(devices, default_out)),
                      (INPUT_DEVICE_UID, input_plan(devices, default_in))):
        name = (plan or {}).get("name", uid)
        if uid in existing:
            log(f"  ✓ {name} already exists")
            continue
        if plan is None:
            log(f"  ! could not build {uid}: no suitable device to pair with BlackHole")
            continue
        members = ", ".join(s["uid"] for s in plan["subdevices"])
        try:
            create_aggregate_device(plan)
            created += 1
            log(f"  ✓ created {name}  [{members}]")
        except Exception as exc:
            log(f"  ! could not create {name}: {exc}")
    return created

def verify_tap() -> tuple[bool, str]:
    """Prove a system-audio tap can actually be created on this machine."""
    try:
        tap_id, _uid = create_system_tap(TAP_NAME + " (setup check)")
    except Exception as exc:
        return False, str(exc)
    destroy_system_tap(tap_id)
    return True, ""

def main() -> int:
    print("== Audio setup for call recording ==")
    version = ".".join(str(p) for p in macos_version()[:2])

    if taps_supported():
        ok, why = verify_tap()
        if ok:
            print(f"  ✓ macOS {version} supports system-audio taps — "
                  "no audio configuration needed.")
            print(f"""
In the Record panel choose "Call audio + my mic (automatic)". It captures both
sides of a call without routing anything: no Multi-Output Device, no BlackHole,
no switching your output. Headphones, speakers, AirPods and swapping between
them mid-call all work, because the tap follows the audio rather than the
device.""".rstrip())
            return 0
        print(f"  ! macOS {version} should support system-audio taps, but creating "
              f"one failed:\n    {why}")
        print("    Falling back to the BlackHole devices below.")
    else:
        print(f"  · macOS {version} is older than 14.2, so system-audio taps are "
              "unavailable. Setting up BlackHole instead.")

    created = ensure_devices()
    print(f"""
Two devices are now available in Audio MIDI Setup:
  {OUTPUT_DEVICE_NAME}  — select as your SYSTEM OUTPUT during a call
      (option-click the volume icon) so you hear the call and BlackHole gets a copy.
  {INPUT_DEVICE_NAME}   — select as the Input in Transcrb's Record panel
      so the recording has both the remote side and your own voice.

On this path, plugging headphones in or out resets the system output, so check the
level meter before you start recording.""".rstrip())
    if created:
        print("\nIf you switch to different headphones later, re-run:\n"
              "  .venv/bin/python -m engine.macos_audio")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
