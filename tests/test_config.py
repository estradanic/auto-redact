"""Config is the single source of defaults; the CLI is generated from it."""

import dataclasses

import pytest

from auto_redact import cli
from auto_redact.config import (
    SECTIONS,
    ClassifyConfig,
    Config,
    RenderConfig,
    SnapshotConfig,
    TrackConfig,
    cache_fields,
    from_namespace,
    parse_overrides,
    resolve,
)


def parse(*argv):
    return cli.build_parser().parse_args(list(argv))


def test_resolve_applies_overrides_on_top_of_a_config():
    base = SnapshotConfig(padding=0.3)
    cfg = resolve(SnapshotConfig, base, {"isolate": False})
    assert (cfg.padding, cfg.isolate) == (0.3, False)
    assert base.isolate is True  # frozen: the original is untouched


def test_resolve_defaults_and_unknown_names():
    assert resolve(RenderConfig, None, {}) == RenderConfig()
    with pytest.raises(TypeError, match="bogus"):
        resolve(RenderConfig, None, {"bogus": 1})


def test_run_command_without_flags_yields_the_default_config():
    args = parse("run", "--video", "v.mp4", "--prompt", "p", "--out", "o.mp4")
    assert from_namespace(args) == Config()


def test_every_cli_visible_field_has_a_flag_on_run():
    run = cli.build_parser()._subparsers._group_actions[0].choices["run"]
    dests = {a.dest for a in run._actions}
    for section, cls in SECTIONS.items():
        for f in dataclasses.fields(cls):
            if f.metadata.get("cli", True):
                assert cli.dest_name(section, f) in dests, (section, f.name)


def test_flags_land_in_the_right_section():
    args = parse(
        "run", "--video", "v", "--prompt", "p", "--out", "o",
        "--stride", "3", "--no-scene-cuts", "--padding", "0.4",
        "--classifier", "siglip", "--threshold", "0.8", "--classify-device", "cpu",
        "--crf", "20", "--encoder", "h264_nvenc", "--no-selective", "--no-resume",
    )
    cfg = from_namespace(args)
    assert (cfg.track.stride, cfg.track.scene_cuts) == (3, False)
    assert cfg.snapshot.padding == 0.4
    assert (cfg.classify.classifier, cfg.classify.threshold, cfg.classify.device) == (
        "siglip", 0.8, "cpu",
    )
    assert (cfg.render.crf, cfg.render.encoder, cfg.render.selective) == (
        20, "h264_nvenc", False,
    )
    assert cfg.resume is False


def test_old_flag_spellings_still_parse():
    args = parse("track", "--video", "v", "--meta", "m", "--sample-rate", "2", "--no-half")
    cfg = from_namespace(args)
    assert (cfg.track.stride, cfg.track.half) == (2, False)


def test_classify_command_accepts_both_device_flags():
    for flag in ("--device", "--classify-device"):
        args = parse("classify", "--meta", "m", "--prompt", "p", flag, "cpu")
        assert from_namespace(args).classify.device == "cpu"


def test_choices_are_enforced():
    with pytest.raises(SystemExit):
        parse("run", "--video", "v", "--prompt", "p", "--out", "o", "--encoder", "nope")


def test_resolved_model_depends_on_the_backend():
    assert ClassifyConfig().resolved_model == "gemma3:4b"
    assert ClassifyConfig(classifier="siglip").resolved_model.startswith("google/siglip")
    assert ClassifyConfig(model="x").resolved_model == "x"


def test_execution_only_settings_are_not_cache_fields():
    fields = cache_fields(ClassifyConfig())
    assert "workers" not in fields and "timeout" not in fields
    assert "think" in fields and "classifier" in fields
    assert "device" not in cache_fields(TrackConfig())
    assert "stride" in cache_fields(TrackConfig())


def test_legacy_yaml_keys_map_onto_the_new_options(tmp_path):
    yaml_file = tmp_path / "c.yaml"
    yaml_file.write_text(
        "prompt: red shirt\nsample_rate: 2\nno_half: true\nno_resume: true\n"
        "track_conf: 0.3\nclassify_workers: 2\n"
    )
    defaults = cli._load_config_file(str(yaml_file))
    assert defaults["stride"] == 2 and defaults["half"] is False
    assert defaults["resume"] is False

    parser = cli.build_parser()
    argv = cli._apply_config_file(
        parser, ["run", "--video", "v", "--out", "o"], defaults
    )
    cfg = from_namespace(parser.parse_args(argv))
    assert argv[argv.index("--prompt") + 1] == "red shirt"
    assert (cfg.track.stride, cfg.track.half, cfg.track.conf) == (2, False, 0.3)
    assert cfg.classify.workers == 2 and cfg.resume is False


def test_command_line_beats_the_config_file(tmp_path):
    yaml_file = tmp_path / "c.yaml"
    yaml_file.write_text("padding: 0.9\n")
    parser = cli.build_parser()
    argv = cli._apply_config_file(
        parser,
        ["run", "--video", "v", "--prompt", "p", "--out", "o", "--padding", "0.1"],
        cli._load_config_file(str(yaml_file)),
    )
    assert from_namespace(parser.parse_args(argv)).snapshot.padding == 0.1


# --- --set and advanced options ---------------------------------------------------


def test_advanced_options_are_hidden_from_help_but_still_work():
    run = cli.build_parser()._subparsers._group_actions[0].choices["run"]
    help_text = run.format_help()
    assert "--head-weight" not in help_text and "--edge-penalty" not in help_text
    assert "--set" in help_text and "--padding" in help_text  # normal options stay
    args = parse("run", "--video", "v", "--prompt", "p", "--out", "o", "--head-weight", "0.1")
    assert from_namespace(args).snapshot.head_weight == 0.1


def test_set_reaches_advanced_options():
    args = parse("run", "--video", "v", "--prompt", "p", "--out", "o",
                 "--set", "head_weight=0.1", "--set", "snapshot.edge_penalty=0.7")
    args.config_overrides = parse_overrides(args.set_options, args.config_sections)
    cfg = from_namespace(args)
    assert (cfg.snapshot.head_weight, cfg.snapshot.edge_penalty) == (0.1, 0.7)


def test_set_reaches_options_with_no_flag_at_all():
    overrides = parse_overrides(["track.imgsz=1280", "imgsz=none"], ("track",))
    assert overrides == {"track": {"imgsz": None}}
    assert parse_overrides(["track.imgsz=1280"], ("track",)) == {"track": {"imgsz": 1280}}


def test_set_coerces_bools_choices_and_cli_spellings():
    got = parse_overrides(
        ["isolate=false", "track_conf=0.3", "classify.on_undecided=keep", "stride=3"],
        ("track", "snapshot", "classify"),
    )
    assert got == {
        "snapshot": {"isolate": False},
        "track": {"conf": 0.3, "stride": 3},
        "classify": {"on_undecided": "keep"},
    }


@pytest.mark.parametrize(
    "bad, message",
    [
        ("nope=1", "unknown option 'nope'"),
        ("head_weight", "NAME=VALUE"),
        ("head_weight=abc", "expected float"),
        ("isolate=maybe", "true or false"),
        ("on_undecided=sometimes", "must be one of"),
    ],
)
def test_set_rejects_bad_input(bad, message):
    with pytest.raises(ValueError, match=message):
        parse_overrides([bad], ("snapshot", "classify"))


def test_set_only_accepts_options_the_command_has():
    with pytest.raises(ValueError, match="unknown option 'head_weight'"):
        parse_overrides(["head_weight=0.1"], ("render",))


def test_set_beats_the_flag_and_the_config_file():
    args = parse("run", "--video", "v", "--prompt", "p", "--out", "o",
                 "--padding", "0.3", "--set", "padding=0.7")
    args.config_overrides = parse_overrides(args.set_options, args.config_sections)
    assert from_namespace(args).snapshot.padding == 0.7


def test_main_reports_a_bad_set_value_as_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["render", "--meta", "m.json", "--out", "o.mp4", "--set", "crf=high",
                  "--no-stats", "--no-inhibit"])
    assert exit_info.value.code == 2
    assert "--set crf" in capsys.readouterr().err


def test_undecided_defaults_to_redact_and_is_a_cli_option():
    assert ClassifyConfig().on_undecided == "redact"
    args = parse("classify", "--meta", "m", "--prompt", "p", "--on-undecided", "error")
    assert from_namespace(args).classify.on_undecided == "error"
