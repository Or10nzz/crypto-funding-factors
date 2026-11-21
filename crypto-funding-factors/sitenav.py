"""Shared top navigation for the research pages served on :8772 (calendar.html, factors.html). Display only."""
PAGES = [("index.html", "总看板"), ("live.html", "实盘"), ("calendar.html", "涨跌日历"), ("factors.html", "因子监控"),
         ("momentum.html", "模拟盘")]
CSS = (".topnav{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 16px}.topnav a{padding:5px 12px;border-radius:999px;"
       "border:1px solid #8885;color:inherit;text-decoration:none;font-size:14px}.topnav a.on{background:#2a6fdb;border-color:#2a6fdb;color:#fff}")


def bar(current):
    return "<nav class='topnav'>" + "".join(
        f"<a href='{h}'{' class=on' if h == current else ''}>{t}</a>" for h, t in PAGES) + "</nav>"
