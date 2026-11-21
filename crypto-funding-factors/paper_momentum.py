#!/usr/bin/env python3
"""Weekly paper trader for the momentum rule (L=30, N=5, U=30). Public data only, no key, no orders.

Each Monday UTC it records the picks for the week, using only finished daily bars for the choice and
that Monday's open as the entry. Results are replayed with momentum.run, the backtest's own code.
Page: paper/momentum.html
"""
import sitenav
import html, json, re, statistics, sys, time, traceback, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
import depth
import glslive, glsrank
import momentum as m
import spike
import p2
import optimize as o

NAV_PAGE = "momentum.html"   # shared top nav (sitenav.py, 2026-11-20)
OUT = Path(__file__).parent / "paper"
STATE = OUT / "momentum.json"
STATE_THU = OUT / "momentum_thu.json"   # Thursday tranche (2026-11-12): base + N6 picks only, no Gc/MGc calls or logs
API = "https://fapi.binance.com/fapi/v1"
L, N, U = 30, 5, 30
D = m.DAY
KIT_FROM = 1795392000000      # Monday 2026-11-23 00:00 UTC: first KIT6 week (registry 2026-11-19 FORWARD TEST KIT6)
KIT_DIR = Path(__file__).parent / "data" / "kit_weekly"
THU_START = 1794441600000     # Thursday 2026-11-12 00:00 UTC: first Thursday tranche record
fmt = lambda t: datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%d")


def get(path, **params):
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "paper-trader"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except OSError:
            if attempt == 3:
                raise
            time.sleep(5)


def monday_of(t):
    d = t - t % D
    while ((d // D) + 3) % 7:
        d -= D
    return d


def thursday_of(t):
    """The latest Thursday 00:00 UTC at or before t (epoch day 0 was a Thursday)."""
    d = t - t % D
    while (d // D) % 7:
        d -= D
    return d


def coin_symbols(fetch=get):
    return sorted(s["symbol"] for s in fetch("exchangeInfo")["symbols"]
                  if s.get("underlyingType") == "COIN" and s["status"] == "TRADING" and s["contractType"] == "PERPETUAL"
                  and re.fullmatch(r"[A-Z0-9]+USDT", s["symbol"]) and s["symbol"] not in m.EXCLUDE)


def bars_of(sym, start, now, fetch=get, hl=None):
    """Daily bars from start. The unfinished current bar is kept only for its open.
    hl: optional dict filled with {open time: (high, low)} of the finished bars (CLV6)."""
    rows, t = {}, start
    while t <= now:
        batch = fetch("klines", symbol=sym, interval="1d", startTime=t, limit=500)
        if not batch:
            break
        for r in batch:
            finished = int(r[0]) + D <= now
            rows[int(r[0])] = (float(r[1]), float(r[4]) if finished else None, float(r[7]) if finished else None)
            if hl is not None and finished:
                hl[int(r[0])] = (float(r[2]), float(r[3]))
        t = int(batch[-1][0]) + D
        if len(batch) < 500:
            break
    return rows


def stable_like(bars):
    px = [v[1] for v in bars.values() if v[1] is not None]
    return bool(px) and 0.97 <= median(px) <= 1.03 and max(px) / min(px) < 1.15


def choose(data, day):
    """Exactly the backtest's choice for this Monday."""
    uni = m.universe(data, day, U)
    s = m.signal(data, uni, day, L)
    if len(uni) < U or len(s) < 2 * N:
        return None
    return [x for _, x in s[-N:]], [x for _, x in s[:N]], uni


# 2026-11-18 PAPER CLEAN-UP (results/registry.txt; user "按建议"): A, D, AD, L14, MGc, N6V2P2 are no longer computed or shown,
# base and N6 are still recorded (the live book's fallback, live_momentum.rule_of) but not shown. Old records stay in the files.
VARIANTS = {}
SHOWN = ["N6V2", "Gc", "LV6", "HIGH6", "CLV6", "BLEND3", "KIT6"]            # what the page shows, in this order
BLEND = ("N6V2", "HIGH6", "CLV6")                                    # BLEND3: one third each, weekly return = their average
LABELS = {"HIGH6": "高点 HIGH6（收盘 ÷ 90 天最高收盘）", "CLV6": "收盘位置 CLV6（30 天平均）", "BLEND3": "三合一 BLEND3（N6V2/HIGH6/CLV6 各 1/3）", "base": "原版", "A": "A 多周期", "D": "D 资金费过滤", "AD": "A+D", "L14": "回看 14 天", "Gc": "反向人数比", "MGc": "动量+反向人数比", "N6": "每边 6 条（实盘 11-16 起）", "N6V2": "每边 6 条 + 插针过滤 V2", "N6V2P2": "N6V2 + 成交额突增过滤 P2", "LV6": "低波动 LV6（做多低波动、做空高波动）", "KIT6": "KiT 模型 KIT6（pred7 前 6 / 后 6）"}
# Weekly gls rules (exam passed 2026-11-11, results/registry.txt): forward test from 2026-11-16. Ranking code shared with the exam (glsrank.py).
GLS_RULES, GLS_MIN, GLS_HTTP = ("Gc", "MGc"), 20, lambda q: glslive.http(q, timeout=8)
GLS_BUDGET_S = 90                      # the base picks (live book) wait for this code: give up on gls after 90 seconds
TRACKED = ["N6V2", "Gc", "LV6", "HIGH6", "CLV6", "KIT6"]
VAR_N = {"N6": 6, "N6V2": 6, "N6V2P2": 6, "LV6": 6, "HIGH6": 6, "CLV6": 6, "KIT6": 6}                  # legs a side, where a tracked version differs from N


def variant_picks(data, uni, day, fetch=get):
    """The other versions being tracked, using the same choice code as the backtest."""
    for sym in (uni if VARIANTS else ()):           # funding was only for the D/AD filters
        pts = funding_of(sym, day - 7 * D, day, fetch)
        pre = [0.0]
        for _, r in pts:
            pre.append(pre[-1] + r)
        data[sym].update(times=[t for t, _ in pts], pre=pre)
    out = {}
    for name, kw in VARIANTS.items():
        longs, shorts = o.choose(data, uni, day, kw["ranker"], kw["fund_filter"])
        out[name] = dict(longs=longs, shorts=shorts)
    try:                                   # the base picks feed the live book: nothing here may stop them
        out.update({k: v for k, v in gls_variants(data, uni, day).items() if k == "Gc"})   # MGc dropped 11-18
    except Exception:
        traceback.print_exc()
    return out


def gls_variants(data, uni, day):
    """Gc and MGc for this Monday from the live gls API; everything used is logged for the later check against
    the official history files (glslive.py). A coin without a value is left out, as in the exam."""
    g, stamps, deadline = {}, {}, time.time() + GLS_BUDGET_S
    for sym in uni:
        r = None
        for attempt in range(2):
            if time.time() > deadline:
                break
            try:
                r = glslive.api_gls(sym, day, GLS_HTTP)
                break
            except Exception:
                time.sleep(1)
        if r:
            stamps[sym], g[sym] = r
        time.sleep(0.05)
    mom = [s for _, s in m.signal(data, uni, day, L)]
    r = glsrank.picks(mom, g, N, GLS_MIN)
    with open(OUT / "gls_picks.jsonl", "a") as f:
        f.write(json.dumps(dict(day=day, recorded_at=int(time.time() * 1000), mom=mom, g=g, stamps=stamps, picks=r)) + "\n")
    print(f"gls: {len(g)}/{len(uni)} coins with a value, picks {'made' if r else 'NOT made'}", flush=True)
    return {k: dict(longs=v[0], shorts=v[1]) for k, v in r.items()} if r else {}


def load_bars(day, now, fetch=get):
    data = {}
    for sym in coin_symbols(fetch):
        hl = {}
        bars = bars_of(sym, day - 125 * D, now, fetch, hl)
        if bars and not stable_like(bars):
            data[sym] = dict(bars=bars, times=[], pre=[0.0], hl=hl)
        time.sleep(0.05)
    return data


def n6_of(data, uni, day):
    """The live book's 6 a side: the same ranking as choose(). None with fewer than 12 coins."""
    s = m.signal(data, uni, day, L)
    return dict(longs=[x for _, x in s[-6:]], shorts=[x for _, x in s[:6]]) if len(s) >= 12 else None


def n6v2_of(data, uni, day):
    """N6 with the spike filter V2 (spike.py, same code as the self-tested backtest). None unless 6 a side."""
    longs, shorts = spike.pick(m.signal(data, uni, day, L), {s: data[s]["bars"] for s in uni}, day, 6)
    return dict(longs=longs, shorts=shorts) if len(longs) == 6 and len(shorts) == 6 else None


def n6v2p2_of(data, day):
    """N6V2 on the P2 universe (p2.py, same code as activefilter.py; p2_selftest.py). None unless 6 a side.
    Forward test pre-registered 2026-11-12 (results/registry.txt "FORWARD TEST PRE-REGISTRATION: P2"), Mondays only."""
    uni = p2.universe(data, day, U)
    longs, shorts = spike.pick(m.signal(data, uni, day, L), {s: data[s]["bars"] for s in uni}, day, 6)
    return dict(longs=longs, shorts=shorts) if len(longs) == 6 and len(shorts) == 6 else None


def retvol(b, day):
    """Sample std of the 7 daily close/close returns ending the day before the open (factorrep.py fac("RETVOL")).
    None unless all 8 closes exist."""
    prev, c = day - D, []
    for i in range(7, -1, -1):
        x = b.get(prev - i * D)
        if not x or x[1] is None or x[1] <= 0:
            return None
        c.append(x[1])
    return statistics.stdev(c[i + 1] / c[i] - 1 for i in range(7))


def lv6_of(data, uni, day):
    """Low volatility LV6: long the 6 lowest RETVOL in the live top 30, short the 6 highest; ties by symbol.
    Forward test pre-registered 2026-11-14 (results/registry.txt "FORWARD TEST PRE-REGISTRATION: low volatility (LV)")."""
    v = sorted((r, s) for s in uni if (r := retvol(data[s]["bars"], day)) is not None)
    return dict(longs=[s for _, s in v[:6]], shorts=[s for _, s in v[-6:]]) if len(v) >= 12 else None


def high90(b, day):
    """HIGH (factorlang "last(c) / max(c, 90)"): close of D-1 over the highest close of the 90 days ending D-1,
    at least 81 of those closes. None otherwise."""
    c = [x[1] for i in range(90) if (x := b.get(day - (i + 1) * D)) and x[1] is not None and x[1] > 0]
    last = b.get(day - D)
    if len(c) < 81 or not last or last[1] is None:
        return None
    return last[1] / max(c)


def clv30(b, hl, day):
    """CLV (factorlang "mean((2*c - h - l)/(h - l), 30)"): mean over the 30 days ending D-1 of (2C-H-L)/(H-L),
    days with H > L only, at least 27 of them. None otherwise."""
    v = []
    for i in range(30):
        t = day - (i + 1) * D
        x, y = b.get(t), hl.get(t)
        if x and y and x[1] is not None and y[0] > y[1]:
            v.append((2 * x[1] - y[0] - y[1]) / (y[0] - y[1]))
    return statistics.fmean(v) if len(v) >= 27 else None


def rank6(vals):
    """Long the 6 highest, short the 6 lowest; ties by symbol. None with fewer than 12 values."""
    v = sorted((r, s) for s, r in vals.items() if r is not None)
    return dict(longs=[s for _, s in v[-6:]][::-1], shorts=[s for _, s in v[:6]]) if len(v) >= 12 else None


def high6_of(data, uni, day):
    """HIGH6 (registry 2026-11-18): top 30, no V2."""
    return rank6({s: high90(data[s]["bars"], day) for s in uni})


def clv6_of(data, uni, day):
    """CLV6 (registry 2026-11-18 PAPER CLEAN-UP): top 30, no V2."""
    return rank6({s: clv30(data[s]["bars"], data[s].get("hl", {}), day) for s in uni})


def kit6_of(week, root=None):
    """KIT6 (registry 2026-11-19): data/kit_weekly/<day>/predictions_<day>.csv from the user's laptop; long the 6 highest
    pred7, short the 6 lowest within that week's universe (ties by symbol), no V2, BTC/ETH allowed. The inputs stop
    before the Monday open, so a file that arrives late is still used. None while the file is missing or short."""
    import csv
    tag = fmt(week["day"])
    f = (root or KIT_DIR) / tag / f"predictions_{tag}.csv"    # 2026-11-19: one name per week (user)
    if not f.exists():
        return None
    uni, vals = set(week["universe"]), {}
    with open(f, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("monday") == tag and r.get("symbol") in uni:
                try:
                    vals[r["symbol"]] = float(r["pred7"])
                except (TypeError, ValueError):
                    pass
    return rank6(vals)


def kit_fill(weeks, now, root=None):
    """Adds KIT6 to every week from KIT_FROM whose predictions have arrived; True when a week changed."""
    changed = False
    for w in weeks:
        if w["day"] >= KIT_FROM and "KIT6" not in w.get("variants", {}):
            try:
                v = kit6_of(w, root)
            except Exception:
                traceback.print_exc()
                v = None
            if v:
                w.setdefault("variants", {})["KIT6"] = dict(v, recorded_at=now)
                changed = True
    return changed


def lv_cross(data, day, now):
    """Once per Monday: (symbol, RETVOL) for the top 100 by 30-day volume (90 days listed) -> paper/lv_cross.jsonl."""
    f = OUT / "lv_cross.jsonl"
    if f.exists() and any(json.loads(x)["day"] == day for x in f.read_text().splitlines() if x.strip()):
        return
    uni = m.universe(data, day, 100)
    cross = [[s, r] for s in uni if (r := retvol(data[s]["bars"], day)) is not None]
    with open(f, "a") as fh:
        fh.write(json.dumps(dict(day=day, recorded_at=now, universe=uni, retvol=cross)) + "\n")


def make_picks(day, now, fetch=get):
    data = load_bars(day, now, fetch)
    try:                                   # research record only: never stops the live book's picks
        lv_cross(data, day, now)
    except Exception:
        traceback.print_exc()
    chosen = choose(data, day)
    if not chosen:
        return None, len(data)
    n6, n6v2 = n6_of(data, chosen[2], day), n6v2_of(data, chosen[2], day)
    variants = variant_picks(data, chosen[2], day, fetch)
    if n6:                                             # fewer coins: no N6 this week (the live book falls back)
        variants["N6"] = n6
    if n6v2:
        variants["N6V2"] = n6v2
    for name, fn in (("LV6", lv6_of), ("HIGH6", high6_of), ("CLV6", clv6_of)):
        try:                                           # research versions: never stop the live book's picks
            v = fn(data, chosen[2], day)
            if v:
                variants[name] = v
        except Exception:
            traceback.print_exc()
    return (*chosen, variants), len(data)


def cycle_thu(now, fetch=get, start=None):
    """Thursday tranche: record the picks for the latest Thursday (base and N6, nothing else)."""
    weeks = json.loads(STATE_THU.read_text()) if STATE_THU.exists() else []
    day = thursday_of(now)
    if (start is None or day >= start) and (not weeks or weeks[-1]["day"] < day):
        data = load_bars(day, now, fetch)
        chosen = choose(data, day)
        if chosen:
            vs = {k: v for k, v in (("N6", n6_of(data, chosen[2], day)), ("N6V2", n6v2_of(data, chosen[2], day))) if v}
            weeks.append(dict(day=day, longs=chosen[0], shorts=chosen[1], universe=chosen[2],
                              variants=vs, recorded_at=now, coins_seen=len(data)))
            tmp = STATE_THU.with_suffix(".tmp")
            tmp.write_text(json.dumps(weeks))
            tmp.replace(STATE_THU)
    return weeks


def funding_of(sym, start, now, fetch=get):
    out, t = [], start
    while t < now:
        batch = fetch("fundingRate", symbol=sym, startTime=t, limit=1000)
        if not batch:
            break
        out += [(int(r["fundingTime"]), float(r["fundingRate"])) for r in batch]
        t = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
    return out


def replay(weeks, now, fetch=get):
    """Results for finished weeks, via the backtest's own run()."""
    done = [w for w in weeks if w["day"] + 7 * D <= monday_of(now)]
    if not done:
        return [], [], {}
    first = done[0]["day"]
    data = {}
    names = {s for w in done for s in w["longs"] + w["shorts"]}
    names |= {s for w in done for v in w.get("variants", {}).values() for s in v["longs"] + v["shorts"]}
    for sym in sorted(names | {"BTCUSDT"}):
        pts = funding_of(sym, first, now, fetch)
        pre = [0.0]
        for _, r in pts:
            pre.append(pre[-1] + r)
        data[sym] = dict(bars=bars_of(sym, first, now, fetch), times=[t for t, _ in pts], pre=pre)
    res = m.run(data, [w["day"] for w in done], None, lambda i: (done[i]["longs"], done[i]["shorts"]), N)
    btc = data["BTCUSDT"]["bars"]
    hold = [btc[w["day"] + 7 * D][0] / btc[w["day"]][0] - 1 for w in done]
    extra = {}
    for name in TRACKED:
        rows = [w for w in done if name in w.get("variants", {})]
        if rows:
            extra[name] = m.run(data, [w["day"] for w in rows], None, lambda i: (rows[i]["variants"][name]["longs"], rows[i]["variants"][name]["shorts"]), VAR_N.get(name, N))
    if all(k in extra for k in BLEND):                  # BLEND3: the weeks all three have, average of their returns
        by = [{r["day"]: r for r in extra[k]} for k in BLEND]
        days = sorted(set.intersection(*(set(b) for b in by)))
        if days:
            extra["BLEND3"] = [dict(day=d, **{f: sum(b[d][f] for b in by) / 3 for f in ("ret", "long", "short", "cost")}) for d in days]
    return res, hold, extra


def write_page(weeks, res, hold, extra, now, error=None):
    e = lambda v: html.escape(str(v))
    cur = weeks[-1] if weeks and weeks[-1]["day"] + 7 * D > monday_of(now) else None
    rows = ""
    eq = hq = 1.0
    btc = {r["day"]: h_ for r, h_ in zip(res, hold)}
    main = extra.get("N6V2", [])                   # the live book's mirror (2026-11-18: base no longer shown)
    for r in main:
        h_ = btc.get(r["day"], 0.0)
        eq *= 1 + r["ret"]
        hq *= 1 + h_
        rows += f"<tr><td>{e(fmt(r['day']))}</td><td>{r['ret']:+.2%}</td><td>{r['long']:+.2%}</td><td>{r['short']:+.2%}</td><td>{h_:+.2%}</td><td>{eq - 1:+.1%}</td><td>{hq - 1:+.1%}</td></tr>"
    x = m.metrics([r["ret"] for r in main]) if main else None
    summary = f"<p>共 {len(main)} 周。累计 {eq - 1:+.1%}，同期持有 BTC {hq - 1:+.1%}，最大回撤 {x['dd']:.1%}。</p>" if x else "<p>还没有走完的一周。</p>"
    hold_now = f"<p><b>{e(fmt(cur['day']))} 这周</b></p>" if cur else f"<p>第一周要到 {e(fmt(next_monday(now)))} 周一才开始（香港时间早上 8 点后）。</p>"
    banner = f"<p><b>最近一次更新失败：{e(error)}</b></p>" if error else ""
    compare = ""
    for name in SHOWN:
        series = extra.get(name)
        if series:
            mt = m.metrics([r["ret"] for r in series])
            compare += f"<tr><td>{e(LABELS[name])}</td><td>{len(series)}</td><td>{mt['final'] - 1:+.1%}</td><td>{mt['dd']:.1%}</td><td>{min(r['ret'] for r in series):+.1%}</td></tr>"
        else:
            compare += f"<tr><td>{e(LABELS[name])}</td><td>0</td><td colspan='3'>{'11-23 起记录' if name in ('HIGH6', 'CLV6', 'BLEND3', 'KIT6') else '还没有走完的一周'}</td></tr>"
    compare = f"<h2>各版本对比</h2><div class='w'><table><tr><th>版本</th><th>周数</th><th>累计</th><th>最大回撤</th><th>最差一周</th></tr>{compare}</table></div>"
    kit_miss = [fmt(w["day"]) for w in weeks if KIT_FROM <= w["day"] and w["day"] + 7 * D <= monday_of(now) and "KIT6" not in w.get("variants", {})]
    compare += f"<p>KIT6 的预测在笔记本上跑，传回来才算；文件晚到也照算（输入只到周一开盘前）。缺的周：{e('、'.join(kit_miss)) if kit_miss else '没有'}。</p>"

    if cur and cur.get("variants"):
        hold_now += "".join(f"<p>{e(LABELS[k])}　做多：{e('、'.join(cur['variants'][k]['longs']))}　做空：{e('、'.join(cur['variants'][k]['shorts']))}</p>"
                            for k in SHOWN if k in cur["variants"])
        if cur["day"] >= KIT_FROM and "KIT6" not in cur["variants"]:
            hold_now += f"<p>{e(LABELS['KIT6'])}　这周还没收到预测（data/kit_weekly/{e(fmt(cur["day"]))}/predictions_{e(fmt(cur["day"]))}.csv）</p>"
    page = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="600"><title>动量模拟盘</title>
<style>:root{{color-scheme:light dark}}body{{max-width:900px;margin:0 auto;padding:24px 16px;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}}.w{{overflow-x:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #8884;padding:6px 8px;text-align:left;white-space:nowrap}}{sitenav.CSS}</style></head><body>{sitenav.bar(NAV_PAGE)}
<h1>相对强弱模拟盘</h1><p>虚拟交易，不下单，不碰钱。每周一按规则换仓，已扣手续费、滑点和资金费率。更新于 {e(datetime.fromtimestamp(now / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M'))} UTC。这条规则没有全部通过回测标准，这里只是在新行情里观察。</p>{banner}
<h2>当前</h2>{hold_now}
<h2>规则</h2><p>都从过去 30 天成交额最大的 30 个币里选，每边 6 条，等额，持有一周，不设止损。N6V2 = 实盘规则（过去 30 天涨幅前 6 / 后 6，加插针过滤）；Gc = 反向人数比；LV6 = 做多低波动、做空高波动；HIGH6 = 离 90 天最高收盘最近的做多、最远的做空；CLV6 = 30 天里收盘常在日内高位的做多、常在低位的做空；BLEND3 = N6V2、HIGH6、CLV6 各三分之一；KIT6 = KiT 模型预测的 7 天收益最高 6 个做多、最低 6 个做空（11-23 起，不进 BLEND3）。2026-11-18 清理：其他版本不再计算和显示，旧记录保留在文件里。</p>
{compare}<h2>N6V2（实盘镜像）每周成绩</h2>{summary}<div class="w"><table><tr><th>周一</th><th>本周</th><th>多单</th><th>空单</th><th>BTC</th><th>策略累计</th><th>BTC 累计</th></tr>{rows}</table></div></body></html>"""
    tmp = OUT / "momentum.tmp"
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(OUT / "momentum.html")
    # weekly results per version, read by alert.py's monthly forward-test report (2026-11-13)
    out = {name: [[r["day"], r["ret"]] for r in series] for name, series in [("base", res)] + list(extra.items()) if series}
    tmp = OUT / "momentum_results.tmp"
    tmp.write_text(json.dumps(dict(labels=LABELS, results=out, shown=SHOWN, at=now)))
    tmp.replace(OUT / "momentum_results.json")


def week_positions(week):
    """Every coin any tracked version holds this week. A coin on both sides keeps the base side."""
    pos = {}
    for v in list(week.get("variants", {}).values()) + [week]:
        pos.update({s: 1 for s in v["longs"]} | {s: -1 for s in v["shorts"]})
    return pos


def daily_depth(weeks, now, fetch=get):
    """One order-book reading per UTC day. A failure here must never stop the trading records."""
    if not weeks or weeks[-1]["day"] + 7 * D <= now:
        return
    marker = OUT / "depth_day.txt"
    today = now - now % D
    if marker.exists() and int(marker.read_text()) >= today:
        return
    try:
        depth.snapshot(week_positions(weeks[-1]), fetch, now, OUT / "depth.jsonl")
        marker.write_text(str(today))
    except Exception:
        traceback.print_exc()


def next_monday(now):
    return monday_of(now) + 7 * D


def save(weeks):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(weeks))
    tmp.replace(STATE)


def cycle(now, fetch=get, start=None):
    weeks = json.loads(STATE.read_text()) if STATE.exists() else []
    day = monday_of(now)
    # the first week is the first Monday after the program started; later weeks are recorded as they come
    begin = start if start is not None else day
    if day >= begin and (not weeks or weeks[-1]["day"] < day):
        chosen, n = make_picks(day, now, fetch)
        if chosen:
            longs, shorts, uni, variants = chosen
            weeks.append(dict(day=day, longs=longs, shorts=shorts, universe=uni, variants=variants, recorded_at=now, coins_seen=n))
            save(weeks)
    if kit_fill(weeks, now):                           # KIT6 predictions come from the laptop, possibly days late
        save(weeks)
    daily_depth(weeks, now, fetch)
    return (weeks, *replay(weeks, now, fetch))


def main():
    now = int(time.time() * 1000)
    start_file = OUT / "momentum_start.txt"
    if not start_file.exists():
        start_file.write_text(str(next_monday(now)))
    start = int(start_file.read_text())
    last = ([], [], [], {})
    while True:
        now = int(time.time() * 1000)
        err = None
        try:
            last = cycle(now, start=start)
        except Exception as ex:
            traceback.print_exc()
            err = f"{type(ex).__name__}: {ex}"
        try:                                     # the Thursday tranche never gets in the way of Monday's picks
            cycle_thu(now, start=THU_START)
        except Exception:
            traceback.print_exc()
        write_page(*last, now, err)
        print(datetime.fromtimestamp(now / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M"), "error" if err else "ok", "weeks", len(last[0]), "finished", len(last[1]), flush=True)
        if "--once" in sys.argv:
            return 1 if err else 0
        time.sleep(300 if err else 3600 - time.time() % 3600 + 120)


if __name__ == "__main__":
    sys.exit(main())
