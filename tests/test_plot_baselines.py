import json

import pytest

from scripts.plot_baselines import CUTOFFS, METRICS, build_figures, load_summary, plot_summary


def summary_file(tmp_path):
    data = {"baselines": {method: {"evaluated_questions": 10,
            "metrics_at_k": {str(k): {metric: value for metric in METRICS} for k in CUTOFFS}}
            for method, value in (("bm25", .4), ("dense", .6))}}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(data))
    return path, data


def test_parsing_and_plotted_values(tmp_path):
    path, data = summary_file(tmp_path)
    assert load_summary(path) == data["baselines"]
    figures = build_figures(load_summary(path))
    import matplotlib.pyplot as plt
    try:
        for name in (m + "_at_k" for m in METRICS):
            ax = figures[name].axes[0]
            assert list(ax.lines[0].get_xdata()) == list(CUTOFFS)
            assert list(ax.lines[0].get_ydata()) == [.4] * 4
            assert list(ax.lines[1].get_ydata()) == [.6] * 4
            assert ax.get_ylim() == (0, 1)
        assert [bar.get_height() for bar in figures["metrics_at_50"].axes[0].patches] == [.4]*3 + [.6]*3
    finally:
        for fig in figures.values(): plt.close(fig)


@pytest.mark.parametrize("fault", ["missing_file", "missing_metric", "nan", "range", "counts", "json"])
def test_clear_errors(tmp_path, fault):
    path, data = summary_file(tmp_path)
    if fault == "missing_file": path = tmp_path / "absent.json"
    elif fault == "missing_metric": del data["baselines"]["dense"]["metrics_at_k"]["10"]["mrr"]
    elif fault == "nan": data["baselines"]["bm25"]["metrics_at_k"]["5"]["recall"] = float("nan")
    elif fault == "range": data["baselines"]["bm25"]["metrics_at_k"]["5"]["recall"] = 2
    elif fault == "counts": data["baselines"]["dense"]["evaluated_questions"] = 9
    if fault != "missing_file": path.write_text("invalid" if fault == "json" else json.dumps(data))
    with pytest.raises(ValueError): load_summary(path)


def test_png_pdf_exports_and_no_overwrite(tmp_path):
    path, _ = summary_file(tmp_path)
    paths = plot_summary(path, tmp_path / "figures")
    assert len(paths) == 8
    from PIL import Image
    for output in paths:
        if output.suffix == ".png":
            with Image.open(output) as image:
                assert image.width >= 1900 and image.height >= 1300
                image.verify()
        else:
            assert output.read_bytes().startswith(b"%PDF-")
    with pytest.raises(ValueError, match="already exist"):
        plot_summary(path, tmp_path / "figures")
