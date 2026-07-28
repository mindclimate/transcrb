"""The audio routing setup a BlackHole install leaves undone.

Installing BlackHole gives you a virtual output and nothing that uses it. The
two devices below are what actually make a call recordable, and creating them by
hand in Audio MIDI Setup is the step everyone skips — which is how a standup
gets recorded as 24 minutes of silence.
"""
import plistlib

from engine.macos_audio import (
    INPUT_DEVICE_NAME,
    INPUT_DEVICE_UID,
    OUTPUT_DEVICE_NAME,
    OUTPUT_DEVICE_UID,
    AudioDevice,
    aggregate_description,
    find_blackhole,
    input_plan,
    output_plan,
)

BLACKHOLE = AudioDevice(uid="BlackHole2ch_UID", name="BlackHole 2ch", inputs=2, outputs=2)
SPEAKERS = AudioDevice(uid="BuiltInSpeakerDevice", name="MacBook Pro Speakers",
                       inputs=0, outputs=2)
HEADPHONES = AudioDevice(uid="AirPods-UID", name="AirPods Pro", inputs=1, outputs=2)
MIC = AudioDevice(uid="BuiltInMicDevice", name="MacBook Pro Microphone", inputs=1, outputs=0)
IPHONE = AudioDevice(uid="iPhoneMic-UID", name="aladin iphone Microphone", inputs=1, outputs=0)

def test_description_uses_the_documented_coreaudio_keys():
    desc = aggregate_description("Name", "com.example.uid",
                                 [("hw-uid", False), ("BlackHole2ch_UID", True)],
                                 stacked=False, master_uid="hw-uid")
    assert desc["name"] == "Name"
    assert desc["uid"] == "com.example.uid"
    assert desc["master"] == "hw-uid"
    assert desc["private"] == 0          # 1 would vanish when the process exits
    assert [s["uid"] for s in desc["subdevices"]] == ["hw-uid", "BlackHole2ch_UID"]

def test_description_survives_a_plist_round_trip():
    # CoreAudio receives this as a parsed plist, so it has to serialise cleanly.
    desc = aggregate_description("N", "u", [("a", False)], stacked=True, master_uid="a")
    assert plistlib.loads(plistlib.dumps(desc)) == desc

def test_stacked_flag_separates_multi_output_from_aggregate():
    multi = aggregate_description("N", "u", [("a", False)], stacked=True, master_uid="a")
    aggregate = aggregate_description("N", "u", [("a", False)], stacked=False, master_uid="a")
    assert multi["stacked"] == 1        # a Multi-Output Device: plays to every member
    assert aggregate["stacked"] == 0    # an Aggregate Device: one combined stream

def test_drift_compensation_is_set_per_member():
    desc = aggregate_description("N", "u", [("hw", False), ("bh", True)],
                                 stacked=True, master_uid="hw")
    drift = {s["uid"]: s["drift"] for s in desc["subdevices"]}
    assert drift == {"hw": 0, "bh": 1}

def test_find_blackhole_matches_on_name():
    assert find_blackhole([SPEAKERS, BLACKHOLE, MIC]) is BLACKHOLE
    assert find_blackhole([SPEAKERS, MIC]) is None

# ---- the Multi-Output Device: hear the call and copy it to BlackHole ----------

def test_output_plan_pairs_the_current_output_with_blackhole():
    plan = output_plan([SPEAKERS, BLACKHOLE, MIC], default_output_uid=SPEAKERS.uid)
    assert plan["name"] == OUTPUT_DEVICE_NAME
    assert plan["uid"] == OUTPUT_DEVICE_UID
    assert plan["stacked"] == 1
    assert [s["uid"] for s in plan["subdevices"]] == [SPEAKERS.uid, BLACKHOLE.uid]

def test_output_plan_clocks_off_the_hardware_and_drifts_blackhole():
    # BlackHole has no physical clock; the real output has to be the master or
    # the two ends of the multi-output slowly slide apart.
    plan = output_plan([HEADPHONES, BLACKHOLE], default_output_uid=HEADPHONES.uid)
    assert plan["master"] == HEADPHONES.uid
    assert {s["uid"]: s["drift"] for s in plan["subdevices"]}[BLACKHOLE.uid] == 1

def test_output_plan_needs_blackhole():
    assert output_plan([SPEAKERS, MIC], default_output_uid=SPEAKERS.uid) is None

def test_output_plan_never_nests_its_own_device():
    # Re-running setup while the Transcrb Multi-Output is the active output must
    # not build a device that contains itself.
    ours = AudioDevice(uid=OUTPUT_DEVICE_UID, name=OUTPUT_DEVICE_NAME, inputs=0, outputs=2)
    plan = output_plan([ours, SPEAKERS, BLACKHOLE], default_output_uid=ours.uid)
    members = [s["uid"] for s in plan["subdevices"]]
    assert OUTPUT_DEVICE_UID not in members
    assert members == [SPEAKERS.uid, BLACKHOLE.uid]   # falls back to real hardware

def test_output_plan_ignores_devices_with_no_output_channels():
    plan = output_plan([MIC, BLACKHOLE, SPEAKERS], default_output_uid=MIC.uid)
    assert MIC.uid not in [s["uid"] for s in plan["subdevices"]]

# ---- the Aggregate Device: capture the call *and* your own voice --------------

def test_input_plan_pairs_the_microphone_with_blackhole():
    plan = input_plan([MIC, BLACKHOLE, SPEAKERS], default_input_uid=MIC.uid)
    assert plan["name"] == INPUT_DEVICE_NAME
    assert plan["uid"] == INPUT_DEVICE_UID
    assert plan["stacked"] == 0
    assert [s["uid"] for s in plan["subdevices"]] == [MIC.uid, BLACKHOLE.uid]

def test_input_plan_does_not_treat_blackhole_as_the_microphone():
    # BlackHole has input channels, so a naive "default input" pick can end up
    # with an aggregate of BlackHole and BlackHole — capturing no voice at all.
    plan = input_plan([BLACKHOLE, MIC], default_input_uid=BLACKHOLE.uid)
    members = [s["uid"] for s in plan["subdevices"]]
    assert members.count(BLACKHOLE.uid) == 1
    assert MIC.uid in members

def test_input_plan_ignores_devices_with_no_input_channels():
    plan = input_plan([SPEAKERS, BLACKHOLE, MIC], default_input_uid=SPEAKERS.uid)
    assert SPEAKERS.uid not in [s["uid"] for s in plan["subdevices"]]

def test_input_plan_prefers_a_built_in_mic_over_a_continuity_iphone():
    # An iPhone drifts in and out of the device list; it is a bad default.
    plan = input_plan([IPHONE, MIC, BLACKHOLE], default_input_uid=IPHONE.uid)
    assert MIC.uid in [s["uid"] for s in plan["subdevices"]]

def test_input_plan_needs_blackhole():
    assert input_plan([MIC, SPEAKERS], default_input_uid=MIC.uid) is None


# ---- system-audio taps: the version that needs no configuration ---------------
# A CoreAudio process tap captures whatever macOS is playing, wherever it is
# playing it. Nothing has to be routed, no output device is switched, and
# changing headphones mid-call does not affect it. This replaces the whole
# BlackHole + Multi-Output dance on macOS 14.2 and newer.

from engine.macos_audio import (
    CAPTURE_DEVICE_NAME,
    CAPTURE_DEVICE_UID,
    capture_description,
    taps_supported,
)

def test_taps_are_supported_from_macos_14_2():
    assert taps_supported((14, 2)) is True
    assert taps_supported((26, 5)) is True
    assert taps_supported((14, 1)) is False
    assert taps_supported((13, 6)) is False

def test_capture_description_combines_the_tap_with_the_microphone():
    desc = capture_description("TAP-UUID", MIC.uid)
    assert desc["name"] == CAPTURE_DEVICE_NAME
    assert desc["uid"] == CAPTURE_DEVICE_UID
    assert [t["uid"] for t in desc["taps"]] == ["TAP-UUID"]
    assert [s["uid"] for s in desc["subdevices"]] == [MIC.uid]

def test_capture_description_clocks_off_the_microphone():
    # The tap has no clock of its own; the mic is the hardware reference.
    desc = capture_description("TAP-UUID", MIC.uid)
    assert desc["master"] == MIC.uid
    assert desc["taps"][0]["drift"] == 1

def test_capture_description_is_not_private_so_ffmpeg_can_open_it():
    # ffmpeg records in a separate process; a private device is invisible to it.
    assert capture_description("TAP-UUID", MIC.uid)["private"] == 0

def test_capture_description_works_with_no_microphone_at_all():
    desc = capture_description("TAP-UUID", None)
    assert desc["subdevices"] == []
    assert [t["uid"] for t in desc["taps"]] == ["TAP-UUID"]
    assert "master" not in desc or desc["master"] == ""

def test_capture_description_survives_a_plist_round_trip():
    desc = capture_description("TAP-UUID", MIC.uid)
    assert plistlib.loads(plistlib.dumps(desc)) == desc
