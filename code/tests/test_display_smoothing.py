"""
Smooth display: the level number and the bars glide between the ~2.2 s updates
instead of stepping. Purely visual - these tests pin that the values rendered are
unchanged and that the CSS that animates them is present.
"""
import app


def _css(monkeypatch):
    captured = []
    monkeypatch.setattr(app.st, 'markdown', lambda body, **k: captured.append(body))
    app.inject_css()
    return "\n".join(captured)


def test_level_block_carries_the_value_for_the_animated_number():
    html = app.render_level_block('MODERATE', 0.3412)
    assert '--num:0.3412' in html                      # drives the counting number
    assert 'aria-label="stress index 0.34"' in html    # the readable value is still there
    assert 'width:34.1%' in html                       # bar keeps sub-percent precision
    assert 'MODERATE' in html


def test_the_animated_number_is_clamped_below_one():
    assert '--num:0.9900' in app.render_level_block('HIGH', 1.0)
    assert '--num:0.0000' in app.render_level_block('LOW', -0.2)


def test_signal_bars_keep_sub_percent_precision():
    html = app.render_signal_rows({'jaw_tension': 0.3456})
    assert 'width:34.6%' in html


def test_css_animates_bars_and_number_over_about_one_update_interval(monkeypatch):
    css = _css(monkeypatch)
    assert '.lvl-bar>i, .sig-bar>i { transition: width 2.2s linear' in css
    assert '@property --num' in css and 'transition: --num 2.2s linear' in css
    assert 'counter-reset: n calc(var(--num) * 100)' in css


def test_the_toolbar_that_holds_the_reopen_sidebar_button_is_never_hidden(monkeypatch):
    """The button that reopens a collapsed sidebar (stExpandSidebarButton) lives inside
    Streamlit's toolbar. Hiding the whole toolbar made it 0x0, so a closed sidebar could
    not be reopened (found on the live site, 2026-10-28)."""
    css = _css(monkeypatch)
    hidden = css.split('{ display:none !important; }')[0]          # the selector list that is hidden
    assert '[data-testid="stToolbar"]' not in hidden
    assert '[data-testid="stToolbarActions"]' in hidden             # only the Deploy/menu actions go
    assert '[data-testid="stExpandSidebarButton"]' in css and 'position:fixed' in css
