"""Build output/Zero_to_One_Findings.pdf from the notebooks' outputs.

Run after any notebook:  python findings_report.py
Each hypothesis section reads its own output/hN_summary.json; sections whose notebook has not
run yet are listed as pending.
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
FIG = OUT / "figures"

# Set to False once 01_H1 has been re-run with the total-assets screen
H1_PROVISIONAL = False

# ---------------------------------------------------------------- fonts and styles
for name, file in [("S", "DejaVuSerif.ttf"), ("SB", "DejaVuSerif-Bold.ttf"), ("SI", "DejaVuSerif-Italic.ttf")]:
    for d in ["/usr/share/fonts/truetype/dejavu/", "C:/Windows/Fonts/"]:
        try:
            pdfmetrics.registerFont(TTFont(name, d + file)); break
        except Exception:
            continue
try:
    pdfmetrics.getFont("S"); BASE, BOLD, ITAL = "S", "SB", "SI"
except Exception:
    BASE, BOLD, ITAL = "Times-Roman", "Times-Bold", "Times-Italic"
from reportlab.pdfbase.pdfmetrics import registerFontFamily
if BASE == "S":
    registerFontFamily("S", normal="S", bold="SB", italic="SI", boldItalic="SB")

INK, MUTED, ACC = colors.HexColor("#2b2b29"), colors.HexColor("#6b6b66"), colors.HexColor("#2a78d6")
WARN = colors.HexColor("#fff4dd")
T = ParagraphStyle("t", fontName=BOLD, fontSize=20, leading=24, textColor=INK, spaceAfter=4)
SUB = ParagraphStyle("s", fontName=ITAL, fontSize=10, leading=13, textColor=MUTED, spaceAfter=12)
H1s = ParagraphStyle("h1", fontName=BOLD, fontSize=14, leading=18, textColor=ACC, spaceBefore=6, spaceAfter=6)
H2s = ParagraphStyle("h2", fontName=BOLD, fontSize=11, leading=14, textColor=INK, spaceBefore=8, spaceAfter=3)
P = ParagraphStyle("p", fontName=BASE, fontSize=9.6, leading=13.4, textColor=INK, spaceAfter=5)
B = ParagraphStyle("b", parent=P, leftIndent=12, bulletIndent=0, spaceAfter=3)
CAP = ParagraphStyle("c", fontName=ITAL, fontSize=8.4, leading=11, textColor=MUTED, spaceAfter=10)
SM = ParagraphStyle("sm", fontName=BASE, fontSize=8.4, leading=11, textColor=INK)


def bullets(items):
    return [Paragraph(x, B, bulletText="•") for x in items]


def fig(name, caption, width=6.6):
    f = FIG / f"{name}.png"
    if not f.exists():
        return [Paragraph(f"[figure {name} not found]", CAP)]
    from PIL import Image as PILImage
    w, h = PILImage.open(f).size
    return [KeepTogether([Image(str(f), width=width * inch, height=width * inch * h / w), Paragraph(caption, CAP)])]


def table(rows, widths, header=True):
    t = Table([[Paragraph(str(c), SM) for c in r] for r in rows], colWidths=[w * inch for w in widths], repeatRows=1)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9d8d2")),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f0ec")))
    t.setStyle(TableStyle(st))
    return t


def note(text):
    t = Table([[Paragraph(text, SM)]], colWidths=[6.7 * inch])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), WARN), ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#eda100")),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    return t


def load(n):
    f = OUT / f"h{n}_summary.json"
    return json.loads(f.read_text()) if f.exists() else None


pct = lambda x, d=0: f"{100 * x:.{d}f}%"

# ---------------------------------------------------------------- H1
def section_h1(s):
    wc, yc = s["wealth_creation"], s["yc"]
    sim = {(r["universe"].split(" ")[0], r["exit_value"]): r for r in s["portfolio_sim"]}
    a0, a1 = s["alpha_top500_base_end"]
    t0, t1 = s["top10_share_base_end"]
    import pandas as pd
    d = pd.read_csv(OUT / "tables" / "h1a_distribution_by_year.csv", index_col=0)
    sig = d[(d.R_pl_vs_lognormal < 0) & (d.p < 0.05)].index.tolist()
    lr = {"neg": int((d.R_pl_vs_lognormal < 0).sum()), "n": len(d),
          "sig_txt": (", ".join(map(str, sig)) + f" (years when the fitted cut-off fell to "
                      f"${d.loc[sig, 'xmin_bn'].min():.0f}–{d.loc[sig, 'xmin_bn'].max():.0f}bn)") if sig else "no year"}
    import re as _re
    clean = lambda n: _re.sub(r"(,? Inc\.?|,? Corp(oration)?\.?|\.com)$", "", n.title().strip()).replace("Nvidia Corp", "Nvidia").replace("Amazon.Com", "Amazon")
    top5 = ", ".join(clean(r["firm"]) for r in s["top5_creators"])
    st = [Paragraph("H1 · The power law (ch. 7)", H1s),
          Paragraph("<b>Claim.</b> “The best investment in a successful fund equals or outperforms the entire rest of the "
                    "fund combined.” More generally, outcomes follow a power law: a few winners produce most of the value.", P),
          Paragraph("<b>Verdict: partly holds.</b> Wealth creation is extremely concentrated, as Thiel says, and has become "
                    "markedly more concentrated since the book. His strong rule, that one holding outweighs all the others, "
                    "describes the best portfolios rather than typical ones.", P)]
    if H1_PROVISIONAL:
        st.append(note("<b>Provisional.</b> The public-company figures in this section come from a run without the "
                       "total-assets screen, which removes filings whose public float is tagged 1,000× too large. They will "
                       "be updated after 01_H1 is re-run. The Y Combinator figures are final."))
        st.append(Spacer(1, 6))
    st += [Paragraph("What the data show", H2s)] + bullets([
        f"<b>A few firms created most of the wealth.</b> Of the {wc['firms_2014']:,.0f} firms that filed with the SEC around "
        f"2014, just <b>{wc['k_all_gain_lower']:.0f} ({wc['k_pct_of_2014_cohort_lower']:.1f}%)</b> account for the cohort's "
        f"entire net gain to {s['years'][1]} if firms that left the data are counted as worthless. If they are counted at "
        f"their 2014 value instead, it takes {wc['k_all_gain_neutral']:.0f} firms ({wc['k_pct_of_2014_cohort_neutral']:.1f}%). "
        f"The ten largest gainers produced {wc['top10_share_of_net_neutral_pct']:.0f}% of the net gain; the top five were {top5}. "
        f"This matches Bessembinder (2018), who found that about 4% of US stocks account for all of the stock market's wealth creation over Treasury bills from 1926 to 2016.",
        f"<b>Most firms did not win.</b> {wc['exits']:,.0f} of the 2014 filers ({pct(wc['exits'] / wc['firms_2014'])}) no longer "
        f"file, through acquisition or failure. Of the {wc['survivors']:,.0f} survivors, {wc['survivor_share_losing_pct']:.0f}% "
        f"are worth less than in 2014.",
        f"<b>The size distribution has a heavy tail, but it is not clearly a power law.</b> Among the 500 largest firms, the "
        f"tail exponent is about {a0:.2f} in 2014 and {a1:.2f} in {s['years'][1]}, close to Zipf's law (α = 2). Against a "
        f"lognormal, the power law fits worse in {lr['neg']} of {lr['n']} years, significantly so (p &lt; 0.05) in "
        f"{lr['sig_txt']}; in the other years the two cannot be told apart. Thiel's argument needs extreme skew, which "
        f"both distributions produce; it does not need an exact power law.",
        ("<b>Has concentration risen since the book? Pending.</b> The 2011–2014 data still contain a few filings tagged "
         "1,000× too large (for example NPS Pharmaceuticals at about $3.5 trillion in 2014), which inflate the pre-2014 "
         "top-10 share and tail. The trend is reported after the re-run with the total-assets screen.") if H1_PROVISIONAL else
        (f"<b>Concentration has risen sharply since the book.</b> The ten largest firms held {pct(t0)} of total public "
         f"float in 2014 and {pct(t1)} in {s['years'][1]}; before the book the share was flat "
         f"({pct(d.loc[2011:2014, 'top10_share'].min())}–{pct(d.loc[2011:2014, 'top10_share'].max())} in 2011–2014). The top "
         f"1% of firms went from {pct(s['top1pct_share_base_end'][0])} to {pct(s['top1pct_share_base_end'][1])}. The tail "
         f"exponent of the 500 largest firms fell from {a0:.2f} to {a1:.2f} (lower means the giants are bigger relative to "
         f"everyone else), a drop of about {(a0 - a1) / d.loc[[2014, s['years'][1]], 'alpha_top500_se'].pow(2).sum() ** 0.5:.1f} "
         f"standard errors. Most of the rise came after 2018."),
        f"<b>The 'best beats the rest' rule holds for successful portfolios only.</b> In simulated 25-stock portfolios of small "
        f"public firms bought in 2014, the best holding outweighed the other 24 combined in "
        f"{pct(sim[('small', 0.0)]['P_best_ge_rest_all'])} of portfolios (exits worth 0) or "
        f"{pct(sim[('small', 1.0)]['P_best_ge_rest_all'])} (exits at cost). Among the top quarter of portfolios by return, "
        f"the share rises to {pct(sim[('small', 0.0)]['P_best_ge_rest_top_quartile'])} and "
        f"{pct(sim[('small', 1.0)]['P_best_ge_rest_top_quartile'])}. For large firms it is only "
        f"{pct(sim[('large', 0.0)]['P_best_ge_rest_all'])}. Thiel's wording, “in a successful fund”, matters.",
        f"<b>Startups show the same pattern.</b> Of YC companies from the 2005–2012 batches, "
        f"{s['yc_outcomes_2005_2012_mean']['Inactive']:.0f}% are inactive, {s['yc_outcomes_2005_2012_mean']['Acquired']:.0f}% "
        f"were acquired, {s['yc_outcomes_2005_2012_mean']['Active']:.0f}% are still private and active, and "
        f"{s['yc_outcomes_2005_2012_mean']['Public']:.1f}% went public. Team sizes above about {yc['xmin']:.0f} staff follow a "
        f"tail with α = {yc['alpha']:.2f}; the top 1% of companies employ {pct(yc['top1pct_share_of_employees'])} of all staff. "
        f"The largest company outweighed the rest of its batch in {yc['share_batches_best_ge_rest'] * yc['batches']:.0f} of "
        f"{yc['batches']:.0f} batches (Stripe, Optimizely, DoorDash, Breadfast). Team size is only a rough stand-in for value, "
        f"and Breadfast's reported 10,000 staff may be out of date.",
    ])
    st += fig("h1_fig3_wealth_creation", "Figure H1-1. Cumulative share of the 2014 cohort's net wealth gain, by firm rank. "
              "The crossing of 100% marks the number of firms that account for all of the net gain.")
    st += fig("h1_fig2_concentration", "Figure H1-2. Left: tail exponent of the 500 largest firms each year, with 95% "
              "confidence band (grey: the free-cut-off fit, for reference). Right: share of total public float held by the top "
              "10 firms and the top 1% of firms.")
    st += fig("h1_fig4_portfolios", "Figure H1-3. Share of portfolio value held by the single best holding, across 20,000 "
              "simulated 25-stock portfolios bought in 2014 (exits valued at 0). Right of the dotted line, the best holding "
              "outweighs the other 24 combined.", width=5.6)
    st += fig("h1_fig6_yc_tail_batches", "Figure H1-4. Left: YC companies by team size (log–log), with the fitted power-law "
              "tail. Right: the largest company's share of its batch's staff; blue bars are batches where it outweighs the rest.")
    st += fig("h1_fig5_yc_outcomes", "Figure H1-5. Status of YC companies by batch year, as recorded in the YC directory.", width=6.0)
    st += [Paragraph("Implications", H2s)] + bullets([
        "For an investor, the skew is the whole game: a portfolio that misses the handful of winners earns little, however "
        "diversified it is. This supports Thiel's advice to judge each investment by whether it could return the whole fund.",
        "The strong rule should not be read as a law of nature. It describes the funds that succeeded, which is partly a "
        "statement about selection: the funds we hear about are the ones that held a winner.",
        "For the AI economy: the five largest wealth creators since 2014 (Nvidia, Microsoft, Apple, Amazon, Alphabet) are all "
        "technology firms, and four of them are among the largest spenders on AI infrastructure. The skew Thiel describes is "
        "already concentrated in the AI supply chain.",
    ])
    st += [Paragraph("Limits", H2s)] + bullets([
        "Public float excludes shares held by insiders, so founder-controlled firms look smaller than they are. A change in "
        "float also reflects share issuance and buybacks, not only returns.",
        "The SEC frames do not say whether a firm that stopped filing was acquired or failed, so H1b and H1c report two bounds.",
        "SEC data begin in 2011, so the comparison before the book covers only four years.",
        "Venture funds are not the public market. Public small caps are the nearest public proxy, and YC is the nearest public "
        "record of an early-stage portfolio.",
    ])
    return st


# ---------------------------------------------------------------- H2
def section_h2(s):
    c = {(r["year"], r["x"]): r for r in s["census_cross_section"]}
    fd = {r["x"]: r for r in s["census_first_diff"]}
    a = s["airlines_vs_bigtech_margin"]
    sh, ld, dm = s["opm_share"], s["opm_leader"], s["opm_dominant"]
    pre, inter = s["opm_share_pre"], s["opm_share_post_interaction"]
    niche, size = s["opm_share_given_size_sector"], s["opm_size_given_share_sector"]
    bins = {b["share_bin"]: b for b in s["share_bins"]}
    ap = lambda y: a[str(y)]
    air_pre = sum(ap(y)["airlines_op_margin_pct"] for y in range(2011, 2015)) / 4
    bt_pre = sum(ap(y)["bigtech_op_margin_pct"] for y in range(2011, 2015)) / 4
    post_years = [y for y in range(2015, 2026) if y != 2020 and str(y) in a]
    air_post = sum(ap(y)["airlines_op_margin_pct"] for y in post_years) / len(post_years)
    bt_post = sum(ap(y)["bigtech_op_margin_pct"] for y in post_years) / len(post_years)
    st = [Paragraph("H2 · “Competition destroys profits” (ch. 3–4)", H1s),
          Paragraph("<b>Claim.</b> “Under perfect competition, in the long run no company makes an economic profit,” while "
                    "“monopoly is the condition of every successful business.” Thiel's example: in 2012 US airlines earned "
                    "about 37 cents per passenger trip, while Google's profit margin was about 21%.", P),
          Paragraph("<b>Verdict: partly holds.</b> A firm that dominates its market earns clearly higher margins, and the "
                    "premium has grown since the book. But industry-wide concentration in manufacturing brings no margin "
                    "premium, and at a given firm size, dominating a small market pays less, not more. What pays is "
                    "dominance at scale.", P),
          Paragraph("What the data show", H2s)] + bullets([
        f"<b>Market leaders earn more.</b> Among {s['n_firms']:,} non-financial SEC filers ({s['n_firm_years']:,} firm-years, "
        f"2011–2025), compared with rivals in the same 4-digit industry and year, ten more points of revenue share goes with "
        f"{10 * sh['coef']:.1f} more points of operating margin (s.e. {10 * sh['se']:.1f}). The industry's largest firm earns "
        f"{100 * ld['coef']:.1f} points more than the rest, and a firm with half the market or more earns {100 * dm['coef']:.1f} "
        f"points more. Return on assets shows the same pattern. Firms under 1% of their market average a "
        f"{100 * bins['<1%']['mean_margin']:.1f}% operating margin; firms above 75% average {100 * bins['>75%']['mean_margin']:.1f}%.",
        f"<b>The premium has grown since the book.</b> The share slope was {pre['coef']:.2f} in 2011–2014 and "
        f"{pre['coef'] + inter['coef']:.2f} in 2015–2025 (the difference has p &lt; 0.001). In Figure H2-1 the post-2014 line "
        f"sits above the pre-2014 line in every band above 1%, and the gap is widest for firms with more than 75% of their market.",
        f"<b>Dominating a niche does not pay at a given size.</b> Comparing firms of the same revenue within the same broad "
        f"sector and year, the one with the larger share of a smaller industry earns slightly less (ten points of share: "
        f"{10 * niche['coef']:+.1f} points of margin), while each doubling of revenue adds about {100 * size['coef'] * 0.693:.1f} "
        f"points. Thiel advises starting with a niche and expanding; the data say the reward comes from the expansion, not "
        f"from the niche.",
        f"<b>Concentrated manufacturing industries do not earn more.</b> Across {c[(2017, 'cr4')]['n']} six-digit manufacturing "
        f"industries in 2017 and {c[(2022, 'cr4')]['n']} in 2022, the price-cost margin is unrelated to the four-firm "
        f"concentration ratio (p = {c[(2017, 'cr4')]['p']:.2f} and {c[(2022, 'cr4')]['p']:.2f}) and slightly <i>lower</i> where "
        f"the HHI is higher. Industries whose concentration rose between 2017 and 2022 saw no margin gain "
        f"(p = {fd['d_cr4']['p']:.2f}).",
        f"<b>Thiel's airline example has held up.</b> The five big tech firms' combined operating margin averaged {bt_pre:.0f}% "
        f"in 2011–2014 and {bt_post:.0f}% in 2015–2025 (excluding 2020), reaching {ap(2025)['bigtech_op_margin_pct']:.0f}% in "
        f"2025. US airlines averaged {air_pre:.0f}% and then {air_post:.0f}%, with a loss of "
        f"{abs(ap(2020)['airlines_op_margin_pct']):.0f}% of revenue in 2020, and stood at {ap(2025)['airlines_op_margin_pct']:.0f}% in 2025.",
    ])
    st += fig("h2_fig2_share_margin", "Figure H2-1. Operating margin relative to the industry-year average, by the firm's "
              "revenue share of its 4-digit SIC industry (median per band). SEC filers excluding financials, 2011–2025.")
    st += fig("h2_fig1_census_margins", "Figure H2-2. Left: manufacturing price-cost margin by HHI (20 bins of equal count). "
              "Right: change in margin against change in four-firm concentration, 2017–2022, one dot per industry.")
    st += fig("h2_fig3_airlines_bigtech", "Figure H2-3. Aggregate operating margin of US scheduled airlines (SIC 4512) and of "
              "Alphabet, Apple, Microsoft, Meta and Amazon, 2011–2025.", width=6.0)
    st += [Paragraph("Implications", H2s)] + bullets([
        "Thiel is right that the winners are the firms that dominate their markets, and that this has become more true since "
        "2014, in line with the rise in concentration found under H1.",
        "He is less right that concentration as such produces profits. In manufacturing, concentrated industries are no more "
        "profitable, which fits the view that a leader's margin reflects its own efficiency and scale (Demsetz's 'efficient "
        "structure' argument) as much as a lack of rivals. These data cannot separate the two.",
        "For AI, the relevant question is whether today's leaders can reach dominance at scale, not whether they hold a niche. "
        "The H1 and H2 results point the same way: the payoff to being the largest firm in a large market has grown.",
    ])
    st += [Paragraph("Limits", H2s)] + bullets([
        "Market share counts only listed rivals; industries with many private firms look more concentrated than they are.",
        "Operating margin is an accounting measure. It ignores the cost of capital and differs with accounting choices, such "
        "as whether R&amp;D is expensed.",
        "SIC codes are broad and are assigned once per firm; a diversified firm such as Amazon is classed in one industry only.",
        "The Census data cover manufacturing only and both years fall after the book, so the industry test says nothing about "
        "change across 2014.",
        "Filings with revenue tagged at the wrong scale (16 firm-years, for example Tigo Energy and Rectitude Holdings) are dropped.",
    ])
    return st


# ---------------------------------------------------------------- H3
def section_h3(s):
    import pandas as pd
    tk = s["top_decile"]
    rs = s["reversion_slopes"]
    lt = {r["dominant"]: r for r in s["leaders_by_type"]}
    lp = {r["period"]: r for r in s["leaders_pre_post"]}
    g = {r["tech"]: r for r in s["payback_by_group"]}
    pb = pd.read_csv(OUT / "tables" / "h3d_cashflow_payback.csv", index_col=0)
    nm = lambda n: n.encode("latin-1", "ignore").decode("utf-8", "ignore").replace("\xa0", " ").title() if isinstance(n, str) else n
    pick = lambda key: pb[pb.firm.str.contains(key, case=False, na=False)].iloc[0]
    apl, goo, meta, msft = pick("Apple"), pick("Alphabet"), pick("Meta Platforms"), pick("Microsoft")
    ge, xom = pick("General Electric"), pick("Exxon")
    surv = pd.DataFrame(s["top_survival"])
    end = s["end_year"]
    st = [Paragraph("H3 · Durability and the last-mover advantage (ch. 5)", H1s),
          Paragraph("<b>Claim.</b> “A great business is defined by its ability to generate cash flows in the future.” Moving "
                    "first matters less than being the <i>last mover</i>, holding a market position that is defensible 10 "
                    "and 20 years out. For a tech company, most of its value lies 10–15 years or more in the future.", P),
          Paragraph("<b>Verdict: holds, with one twist.</b> Leading positions are durable, and more so since the book; "
                    "dominant leaders keep their markets three times in four. Profit advantages fade, but slowly. The twist "
                    "is in prices: in 2014 the market did <i>not</i> price the tech giants as if their value lay far in the "
                    "future. Their next 11 years of cash flow repaid the whole 2014 price, so investors underrated exactly "
                    "the durability Thiel describes.", P),
          Paragraph("What the data show", H2s)] + bullets([
        f"<b>Profitability is sticky but not permanent.</b> Of firms in the top tenth by return on assets, "
        f"{pct(tk['base 2011-2014|k=1']['still_top_decile'])} are still there a year later, "
        f"{pct(tk['base 2011-2014|k=5']['still_top_decile'])} after five years and "
        f"{pct(tk['base 2011-2014|k=10']['still_top_decile'])} after ten, while "
        f"{pct(tk['base 2011-2014|k=10']['gone'])} stop filing within ten years. About "
        f"{pct(1 - rs['base 2011-2014|k=1'][0])} of a firm's profitability gap closes each year, a little slower than "
        f"Fama and French's (2000) estimate of about 38%. A third of the gap ({rs['base 2011-2014|k=10'][0]:.2f}) is still there "
        f"after ten years. The rate is the same for base years before and after 2014.",
        f"<b>Leaders keep their markets, and more so since the book.</b> Of the largest firms in each 4-digit industry, "
        f"{lp['base 2011-2014']['still leader']:.0f}% of those from base years 2011–2014 still led five years later, against "
        f"{lp['base 2015-2020']['still leader']:.0f}% from 2015–2020. The share that disappeared fell from "
        f"{lp['base 2011-2014']['gone']:.0f}% to {lp['base 2015-2020']['gone']:.0f}%. Leaders with half their market or more "
        f"kept the lead {lt['True']['still leader']:.0f}% of the time, against {lt['False']['still leader']:.0f}% for other leaders.",
        f"<b>Yet the very top turns over.</b> Only {s['top20_2014_still']} of the 20 largest US firms of 2014 are in the top 20 "
        f"in {end}: Apple, Microsoft, Alphabet, Berkshire Hathaway, JPMorgan, Exxon Mobil, Johnson &amp; Johnson and Bank of "
        f"America. General Electric, Pfizer, Intel, Verizon and SLB fell out, Intel and SLB below rank 100. Twelve firms "
        f"replaced them, led by Nvidia, Amazon, Meta, Broadcom and Tesla. In any five-year window about "
        f"{pct(surv.top20_survive.mean())} of the top 20 stay in the top 20.",
        f"<b>The winners match Thiel's playbook.</b> The newcomers are mostly firms that began by dominating a narrow market and "
        f"expanded (Nvidia from gaming graphics chips, Amazon from books, Meta from one university), while the leavers include "
        f"first movers that were overtaken (Intel, IBM) and the energy and conglomerate giants of 2014.",
        f"<b>2014 prices undervalued durable tech firms.</b> For {len(pb)} of the 100 largest non-financial firms, the present "
        f"value of 2015–{end} free cash flow (8% discount rate) covered a median {pct(g['tech']['pv_payback'])} of the 2014 "
        f"price for tech firms and {pct(g['other']['pv_payback'])} for the rest. Apple ({pct(apl.pv_payback)}), Meta "
        f"({pct(meta.pv_payback)}) and Alphabet ({pct(goo.pv_payback)}) repaid their entire 2014 price within 11 years, and "
        f"their public float still grew {apl.value_multiple:.0f}×, {meta.value_multiple:.0f}× and {goo.value_multiple:.1f}×. "
        f"Microsoft repaid {pct(msft.pv_payback)} and grew {msft.value_multiple:.1f}×. The prices that leaned most on cash "
        f"flows after {end} were those of mature firms: General Electric ({pct(ge.terminal_share)} of its 2014 price) and "
        f"Exxon Mobil ({pct(xom.terminal_share)}), whose values were flat over the period.",
    ])
    st += fig("h3_fig2_leader_persistence", "Figure H3-1. Industry leaders (largest firm by revenue in each 4-digit SIC "
              "industry with at least 3 SEC filers), five years later.", width=5.8)
    st += fig("h3_fig1_roa_persistence", "Figure H3-2. Left: share of top-decile ROA firms still in the top decile, and share "
              "no longer filing, k years later. Right: slope of a firm's ROA gap (relative to the yearly median) at t+k on its "
              "gap at t, with 95% intervals.")
    st += fig("h3_fig3_top20", f"Figure H3-3. The 20 largest US firms by public float in 2014 and their rank in {end}.", width=6.2)
    st += fig("h3_fig4_payback", "Figure H3-4. Present value of 2015–2025 free cash flow as a share of 2014 public float, "
              "30 largest non-financial firms. Blue: tech (SIC 357x, 3661–3679, 48xx, 737x).", width=5.6)
    st += [Paragraph("How this compares with the book notes", H2s)] + bullets([
        "<i>“Entrepreneurs should think of being the last mover in a particular market.”</i> Supported. Leadership is more "
        "persistent than profitability, and dominant leaders hold on best.",
        "<i>Durability question: “Will your market position be defensible 10 and 20 years into the future?”</i> Mostly "
        "supported. A third of a profit advantage survives ten years, but only 40% of the top 20 survived eleven years at the "
        "very top. Durability is the exception that pays, not the rule.",
        "<i>“Most of a tech company's value comes 10–15 years out.”</i> True for a young firm such as PayPal in 2001, but for "
        "the established tech monopolies of 2014 the market priced too little durability, not too much. Thiel's point that "
        "investors fixate on near-term growth and undervalue durable cash flows fits this better than the literal claim.",
        "<i>The shale example in the durability question.</i> The energy majors that dominated 2014 (Exxon, Chevron, SLB) all "
        "lost rank, which fits his warning that energy positions can be undercut quickly.",
    ])
    st += [Paragraph("Limits", H2s)] + bullets([
        "Public float excludes insider stakes (large at Meta, Amazon and Walmart), which makes their 2014 values look smaller "
        "and their payback ratios higher than full market values would.",
        "Spin-offs are not followed: DuPont's 2014 value includes Dow and Corteva, which it later spun off, and General Electric "
        "spun off GE HealthCare and GE Vernova. Their payback and value multiples are understated.",
        "Free cash flow ignores acquisitions and treats stock-based compensation as costless.",
        "The 'tech' group follows SIC codes, so it includes telecoms (Verizon, AT&amp;T, Comcast) and leaves out Amazon "
        "(retail) and Visa (business services).",
        "Ten-year tests have only five base years (2011–2015).",
    ])
    return st


# ---------------------------------------------------------------- H4
def section_h4(s):
    sc = {r["ticker"]: r for r in s["scorecard"]}
    gp = {r["name"]: r for r in s["groups"]}
    cash = {r["ticker"]: r for r in s["deals_cash_alternative"]}
    f = lambda t: float(sc[t]["multiple"])
    v = lambda t: float(sc[t]["vs_spy"])
    dd = lambda t: abs(float(sc[t]["max_drawdown_pct"]))
    mono = gp["Monopolies and definite planners"]
    mono_t = ["TSLA", "AMZN", "AAPL", "MSFT", "GOOGL", "EBAY+PYPL", "META"]
    nm = {"TSLA": "Tesla", "AMZN": "Amazon", "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
          "EBAY+PYPL": "eBay with PayPal", "META": "Meta"}
    end = s["window"][1]
    st = [Paragraph("H4 · The companies Thiel named, after the book (ch. 3, 5, 6, 12, 13)", H1s),
          Paragraph("<b>Question.</b> A reader who bought the book's examples on publication day (16 September 2014) and "
                    f"held them to {end}: who won? Total returns with dividends reinvested, against the S&amp;P 500. "
                    "Acquired firms are valued at their deal terms, with the proceeds reinvested in the S&amp;P 500.", P),
          Paragraph("<b>Verdict: mostly holds.</b> Every firm Thiel held up as a monopoly or a definite planner beat the "
                    "index, and his two warnings about competitive or faddish businesses (airlines, Groupon) came true. "
                    "His chapter 5 valuation point reversed: the New York Times, which he contrasted unfavourably with "
                    "Twitter, earned three times as much as Twitter's holders. Zynga, bought after its collapse, ended "
                    "ahead of the index.", P),
          Paragraph("What the data show", H2s)] + bullets([
        f"<b>The monopolies won, every one.</b> An equal-weight portfolio of the seven returned {float(mono['multiple']):.1f}× "
        f"({float(mono['cagr_pct']):.1f}% a year) against {f('SPY'):.1f}× for the S&amp;P 500 and {f('QQQ'):.1f}× for the "
        f"Nasdaq-100. By company: " + ", ".join(f"{nm[t]} {f(t):.1f}×" for t in mono_t) + ". None trailed the index. "
        f"The ride was rough: Meta and eBay with PayPal each fell {dd('META'):.0f}% from a high at some point.",
        f"<b>Competition did destroy returns.</b> Six US airlines returned {f('AIRLINES'):.1f}× in total, under a third of the "
        f"index, with a {dd('AIRLINES'):.0f}% drawdown in 2020. Groupon lost {100 * (1 - f('GRPN')):.0f}% of its value "
        f"({f('GRPN'):.2f}×), falling {dd('GRPN'):.0f}% at the worst.",
        f"<b>The 'distant cash flows' firms trailed.</b> Microsoft bought LinkedIn in December 2016 for $196 a share, below "
        f"its $212.38 price on publication day, so holders got {float(cash['LNKD']['multiple']):.2f}× back after two years. "
        f"Twitter went private in October 2022 at $54.20 against $50.83, which is {float(cash['TWTR']['multiple']):.2f}× after "
        f"eight years. With the proceeds reinvested in the index, they end at {f('LNKD'):.1f}× and {f('TWTR'):.1f}×, both "
        f"behind the S&amp;P 500. The future cash flows the market priced in 2014 never reached their shareholders.",
        f"<b>The New York Times beat Twitter and the index.</b> Thiel noted that loss-making Twitter was valued above the "
        f"profitable Times. Since then the Times returned {f('NYT'):.1f}×, against {f('TWTR'):.1f}× for Twitter and "
        f"{f('SPY'):.1f}× for the index. The Times built a durable subscription business, so the outcome fits Thiel's "
        f"durability test even as it overturns his example.",
        f"<b>Zynga is the exception in its group.</b> Bought at $3.02, already about 70% below its 2011 IPO price, it was "
        f"acquired by Take-Two in May 2022 for $3.50 cash plus 0.0406 Take-Two shares (about $8.52). With the proceeds "
        f"reinvested it ends at {f('ZNGA'):.1f}×, ahead of the index; held as cash and Take-Two shares, "
        f"{float(cash['ZNGA']['multiple']):.1f}×. A doubtful-durability firm can still pay off when bought cheaply.",
        f"<b>Palantir</b>, which Thiel co-founded and uses in chapter 12, is up {f('PLTR'):.1f}× since its September 2020 "
        f"listing ({v('PLTR'):.1f}× the index over the same window), after an {dd('PLTR'):.0f}% fall along the way.",
    ])
    st += fig("h4_fig1_groups", "Figure H4-1. Value of $1 invested on 16 September 2014 in each group (equal weight, log "
              "scale), against the S&amp;P 500 total return.")
    st += fig("h4_fig2_vs_spy", "Figure H4-2. Each company's multiple relative to the S&amp;P 500 over the same window "
              "(log scale); the right-hand column shows what $1 became.", width=6.0)
    st += fig("h4_fig3_pairs", "Figure H4-3. The book's two direct comparisons: Twitter against the New York Times (ch. 5) "
              "and Google against the airlines (ch. 3).")
    st += [Paragraph("How this compares with the book notes", H2s)] + bullets([
        "<i>“Don't build an undifferentiated commodity business.”</i> The airlines are the clearest case: a decade of "
        "growing demand, and shareholders earned almost nothing.",
        "<i>“Competition and capitalism are opposites.”</i> The firms Thiel called monopolies "
        "delivered roughly three times the index. This also matches H1 and H2: they are among the handful of firms that "
        "created most of the decade's wealth.",
        "<i>“Think about what you do and the future value it holds.”</i> LinkedIn and Twitter show the risk on the buyer's "
        "side. A price that depends on distant cash flows pays off only if the business becomes durable, and neither did "
        "so while public.",
    ])
    st += [Paragraph("Limits", H2s)] + bullets([
        "This is a scorecard of fourteen stocks over one decade, not a test of cause. The book's examples were chosen with "
        "hindsight to fit its argument, and the monopoly group is also the mega-cap technology group that led the whole "
        "market.",
        "Daily prices for LinkedIn, Twitter and Zynga come from public copies of Yahoo's history, because Yahoo no longer "
        "carries them. Their publication-day closes match several independent copies.",
        "eBay with PayPal is measured on prices without dividends; eBay's dividend began only in 2019.",
        "Reinvesting buyout proceeds in the index is a choice. Holding them as cash would lower LinkedIn and Twitter "
        "further.",
    ])
    return st


# ---------------------------------------------------------------- H5
def section_h5(s):
    import pandas as pd
    era = {r["era"]: r for r in s["eras"]}
    t16 = {(r["profitability"], r["sales_group"]): r for r in s["ritter_t16b"]}
    so = s["sec_outcomes"]; sz = s["sec_outcomes_by_size"]
    L, Pf = so["Losing money"], so["Profitable"]
    sol = {int(k): v for k, v in s["solar_twh"].items()}
    ct = pd.DataFrame(s["cleantech"])
    bk = ct[ct.window.str.startswith("since book")].set_index("ticker")
    spy = bk.loc["SPY", "multiple"]
    names = bk.drop(index=["SPY", "QQQ"])
    losers = names[names.multiple < 1.0001]
    has07 = ct[ct.window.str.startswith("since end-2007")].start.astype(str).str[:4].isin(["2007", "2008"]).any()
    nm = {"FSLR": "First Solar", "QCLN": "QCLN", "ICLN": "ICLN", "ENPH": "Enphase", "SEDG": "SolarEdge", "TAN": "TAN",
          "PBW": "PBW", "RUN": "Sunrun", "JKS": "JinkoSolar", "CSIQ": "Canadian Solar"}
    st = [Paragraph("H5 · Indefinite optimism and companies without plans (ch. 2, 6, 13)", H1s),
          Paragraph("<b>Claim.</b> Since the 1980s American finance has been <i>indefinitely optimistic</i>: money flows to "
                    "companies on the general hope that the future will be better, not on a definite plan (ch. 6). “A bad "
                    "plan is better than no plan” (ch. 2). Cleantech showed the result: investors bet on the general idea of "
                    "green energy and funded firms without specific plans, and the outcome was a bubble (ch. 13).", P),
          Paragraph("<b>Verdict: holds.</b> The pattern Thiel described got stronger after the book. Money-losing IPOs "
                    "reached their highest sustained share on record, and a SPAC wave followed. Firms that listed while "
                    "losing money rarely became profitable and earned their investors less. In cleantech, the technology "
                    "triumphed while its investors trailed the market, which is the clearest case in the whole study.", P),
          Paragraph("What the data show", H2s)] + bullets([
        f"<b>Loss-making IPOs became the norm after the book.</b> Ritter's data show {era['2014-2021 (after the book)']['pct_unprofitable']:.0f}% "
        f"of US IPOs in 2014–2021 had losses in the year before listing, against {era['1995-2000 (dot-com)']['pct_unprofitable']:.0f}% in "
        f"1995–2000 and {era['1980-1994']['pct_unprofitable']:.0f}% in 1980–1994. The share was 75% or higher in every year from "
        f"2017 to 2022, a level previously reached only in 1999 and 2000. SPAC listings, shells that raise money before they "
        f"have chosen a business, rose to 248 in 2020 and 613 in 2021, raising {era['2014-2021 (after the book)']['spac_proceeds_bn']:.0f} "
        f"billion dollars in 2014–2021.",
        f"<b>Our own count confirms it.</b> Of {s['sec_cohorts_n']:,} firms that began filing with the SEC in 2013–2022 (SPACs "
        f"excluded), {s['sec_pct_loss']:.0f}% were losing money on arrival. Year by year the share tracks Ritter's IPO figures "
        f"(for example 72% against 71% in 2014, and 81% against 80% in 2020).",
        f"<b>Few loss-makers turned a profit.</b> Three years after arriving, {L['profitable_after_3y_pct']:.0f}% of the "
        f"loss-makers were profitable, against {Pf['profitable_after_3y_pct']:.0f}% of firms that arrived profitable. "
        f"{L['survived_3y_pct']:.0f}% were still listed (against {Pf['survived_3y_pct']:.0f}%), and the median survivor's value "
        f"rose {100 * (L['median_value_multiple_survivors'] - 1):.0f}% (against {100 * (Pf['median_value_multiple_survivors'] - 1):.0f}%). "
        f"Among firms with revenue under $100 million, only {sz['Revenue < $100m|Losing money']['profitable_after_3y_pct']:.0f}% "
        f"of loss-makers were profitable three years on.",
        f"<b>Over 45 years, investors paid for it.</b> In Ritter's data for 1980–2024, the three-year return of loss-making "
        f"IPOs trailed that of similar listed firms by {abs(t16[('Unprofitable', 'All')]['bhr3_style_adj_pct']):.1f}%, while "
        f"profitable IPOs beat them by {t16[('Profitable', 'All')]['bhr3_style_adj_pct']:.1f}%. The gap comes from small firms: "
        f"loss-makers with sales under $100 million trailed by {abs(t16[('Unprofitable', 'Sales < $100m')]['bhr3_style_adj_pct']):.1f}%, "
        f"large ones by only {abs(t16[('Unprofitable', 'Sales > $100m')]['bhr3_style_adj_pct']):.1f}%.",
        f"<b>Cleantech: the technology won, its investors did not.</b> World solar generation rose from {sol[2014]:.0f} TWh in "
        f"2014 to {sol[max(sol)]:,.0f} TWh in {max(sol)}, a {sol[max(sol)] / sol[2014]:.0f}-fold increase. Yet all {len(names)} "
        f"clean-energy funds and companies tested trailed the S&amp;P 500 ({spy:.1f}×) from the book's date. The best was "
        f"First Solar at {bk.loc['FSLR', 'multiple']:.1f}×. The solar fund TAN returned {bk.loc['TAN', 'multiple']:.2f}× and "
        f"PBW, the cleantech fund used in notebook 01, {bk.loc['PBW', 'multiple']:.2f}×. "
        + (f"{len(losers)} of the {len(names)} lost money outright: " + ", ".join(f"{nm[t]} ({m:.2f}×)" for t, m in losers.multiple.sort_values().items()) + "." if len(losers) else ""),
    ])
    st += fig("h5_fig1_unprofitable_ipos", "Figure H5-1. Left: share of US IPOs with negative earnings before listing, "
              "1980–2025. Right: operating-company and SPAC IPOs per year. Source: Ritter, Tables 9 and 15b.")
    st += fig("h5_fig3_sec_cohorts", "Figure H5-2. Left: share of new SEC filers (2013–2022 cohorts, SPACs excluded) losing "
              "money on arrival, against Ritter's IPO series. Right: outcomes three years later.")
    st += fig("h5_fig2_ritter_returns", "Figure H5-3. Average three-year return after IPO relative to firms of similar size "
              "and book-to-market, by pre-IPO profitability and sales, IPOs of 1980–2024. Source: Ritter, Table 16b.", width=5.6)
    st += fig("h5_fig4_cleantech", "Figure H5-4. Left: world solar generation and two clean-energy funds, indexed to the end "
              "of 2014. Right: each clean-energy fund or company against the S&amp;P 500 since 16 September 2014 (SolarEdge and "
              "Sunrun from their 2015 listings).")
    st += [Paragraph("How this compares with the book notes", H2s)] + bullets([
        "<i>“When indefinitely optimistic investors betting on the general idea of green energy funded cleantech companies "
        "that lacked specific business plans, the result was a bubble.”</i> The bubble deflated, as notebook 01 showed, and the "
        "decade since confirms the deeper point. Solar behaved like a commodity: output soared, but the firms making it could "
        "not keep the gains. The high-volume manufacturers producing mainly in China (JinkoSolar, Canadian Solar) did worst for "
        "shareholders, which fits Thiel's chapter 3 argument that competition destroys profits.",
        "<i>“Indefinite optimism”</i> as a description of the capital markets fits the post-2014 decade better than the decade "
        "before it: record loss-making IPO shares, and the SPAC boom of 2020–21.",
        "<i>“A bad plan is better than no plan.”</i> The data cannot see plans directly. Losses before listing are a proxy, "
        "and an imperfect one: Amazon and Tesla lost money for years by design. The large-firm split in Ritter's table shows "
        "that losses matter far less for established businesses with a real product.",
    ])
    st += [Paragraph("Limits", H2s)] + bullets([
        "Ritter's tables are transcribed from his PDF into the project's reference folder; each file names its table and date.",
        "New SEC filers include uplistings, spin-offs and firms that merged with SPACs, not only IPOs.",
        "Value multiples use public float, which also moves with share issuance; loss-making firms issue more shares.",
        ("" if has07 else "The comparison from the 2007 cleantech peak will be added after a re-run: the first run's price "
         "data started in 2010."),
    ])
    return [x for x in st if not (hasattr(x, "text") and x.text == "")]


# ---------------------------------------------------------------- H6
def section_h6(s):
    pr = {(r["measure"][:6], r["period"]): r["growth_pct_per_year"] for r in s["productivity"]}
    lp = {k[1]: v for k, v in pr.items() if k[0] == "Labour"}
    tf = {k[1]: v for k, v in pr.items() if k[0] == "Total "}
    rec = {r["period"]: r["growth_pct_per_year"] for r in s["productivity_recent"]}
    ex = s["productivity_post2014_ex_covid"][0]["growth_pct_per_year"]
    ga = s["group_avg"]
    b = {r["item"]: r for r in s["bits_vs_atoms"]}
    inv = s["investment_shares"]
    le = {r["period"]: r["years_gained"] for r in s["life_expectancy"]}
    end = max(int(k[-4:]) for k in lp)
    kpost = [k for k in lp if k.startswith("2014")][0]
    tpost = [k for k in tf if k.startswith("2014")][0]
    tfirst = [k for k in tf if k.startswith("195")][0]
    ib = list(inv)[-1]
    E = [k for k in b["Televisions"] if k.startswith("real_change_2014")][0]
    st = [Paragraph("H6 · Stagnation: progress in bits, not atoms (ch. 1 and conclusion)", H1s),
          Paragraph("<b>Claim.</b> Outside computers and communications, progress has slowed sharply since the early "
                    "1970s. “Technology” has come to mean information technology, while energy, transport, medicine and "
                    "construction have stalled. The book closes by asking whether the future holds stagnation or takeoff.", P),
          Paragraph("<b>Verdict: mostly holds.</b> The split between bits and atoms is stark and has continued since the "
                    "book. Productivity growth has not returned to its 1948–73 pace, and American life expectancy has gone "
                    "nowhere for a decade. Since 2022 productivity has picked up, which is the first sign of the takeoff "
                    "Thiel hoped for, but it is too early to call it one.", P),
          Paragraph("What the data show", H2s)] + bullets([
        f"<b>Productivity growth never recovered its post-war pace.</b> Output per hour in the nonfarm business sector grew "
        f"{lp['1948-1973']:.1f}% a year in 1948–73, then {lp['1973-1995']:.1f}% in 1973–95. The IT boom lifted it to "
        f"{lp['1995-2004']:.1f}% in 1995–2004 before it fell back to {lp['2004-2014']:.1f}% in 2004–14. Since the book it has "
        f"averaged {lp[kpost]:.1f}% ({ex:.1f}% if the distorted pandemic years 2020 and 2021 are left out). Total factor "
        f"productivity, the part not explained by more capital or labour, tells the same story more sharply: "
        f"{tf[tfirst]:.1f}% a year in {tfirst}, {tf['1973-1995']:.1f}% in 1973–95, {tf['1995-2004']:.1f}% in 1995–2004, "
        f"{tf['2004-2014']:.1f}% in 2004–14 and {tf[tpost]:.1f}% in {tpost}.",
        f"<b>A possible turn since 2022.</b> Labour productivity has grown {rec.get('2022-' + str(end), float('nan')):.1f}% a "
        f"year since 2022, faster than its average in 2004–14 or in the full period since the book. It coincides with the spread of generative AI, "
        f"but also with the recovery from the pandemic; three years cannot separate the two.",
        f"<b>Bits keep getting cheaper; atoms do not.</b> Relative to all consumer prices, the four information goods fell "
        f"{abs(ga['bits']['real_change_2000_2014_pct_yr']):.0f}% a year on average in 2000–14 and "
        f"{abs(ga['bits'][E]):.0f}% a year in 2014–{end}. A television now costs about 1% of its 2000 price relative to other "
        f"goods; computers about 4%. The seven physical goods and services rose {ga['atoms']['real_change_2000_2014_pct_yr']:.1f}% "
        f"a year relative to the CPI in 2000–14 and fell {abs(ga['atoms'][E]):.1f}% a year since, so the gap narrowed only "
        f"slightly.",
        f"<b>Some atoms moved after 2014.</b> Airfares fell {abs(b['Airline fares']['cumulative_2014_' + str(end) + '_pct']):.0f}% "
        f"relative to the CPI between 2014 and {end}, and published college tuition, which rose "
        f"{b['College tuition and fees']['real_change_2000_2014_pct_yr']:.1f}% a year faster than prices in 2000–14, has fallen "
        f"slightly in real terms since. Rent did the opposite, rising {b['Rent of primary residence']['cumulative_2014_' + str(end) + '_pct']:.0f}% "
        f"faster than prices since 2014.",
        f"<b>Investment followed the bits.</b> Software, R&amp;D and other intellectual property took "
        f"{inv['1950']['Intellectual property products']:.0f}% of business investment in 1950, "
        f"{inv['2014']['Intellectual property products']:.0f}% in 2014 and {inv[ib]['Intellectual property products']:.0f}% "
        f"in {ib}, overtaking equipment ({inv[ib]['Equipment']:.0f}%). Buildings fell from "
        f"{inv['1950']['Structures']:.0f}% to {inv[ib]['Structures']:.0f}%.",
        f"<b>Medicine stalled.</b> US life expectancy at birth rose 1–3 years every decade from 1960 to 2010. It rose only "
        f"{le['2010-2019']:.2f} years in 2010–19, was flat in 2014–19 ({le['2014-2019']:+.2f}), fell {abs(le['2019-2021']):.1f} "
        f"years in the pandemic and has since recovered; in {s['le_latest'][0]} it stood at {s['le_latest'][1]:.1f}, "
        f"{le.get('2014-' + str(s['le_latest'][0]), 0):+.2f} years against 2014.",
    ])
    st += fig("h6_fig1_productivity", "Figure H6-1. Average annual growth of US labour productivity (nonfarm business, BLS) "
              "and total factor productivity (Penn World Table), by period.")
    st += fig("h6_fig2_bits_atoms", "Figure H6-2. Consumer prices relative to the all-items CPI, 2000 = 100 (log scale). "
              "Blue: information goods; orange: physical goods and services. BLS data via FRED and the BLS API.")
    st += fig("h6_fig3_investment", "Figure H6-3. Shares of US private nonresidential fixed investment, nominal (BEA).", width=6.0)
    st += fig("h6_fig4_life_expectancy", "Figure H6-4. US life expectancy at birth (World Bank).", width=6.0)
    st += [Paragraph("How this compares with the book notes", H2s)] + bullets([
        "<i>The four trends that erased belief in secrets: incrementalism, risk aversion, complacency and flatness.</i> "
        "The productivity record is consistent with a society that stopped making big physical "
        "bets: the fast decades were 1948–73 and the IT boom, not the years since.",
        "<i>“There are a few plausible scenarios for the future: plateau, recurrent stagnation, extinction, takeoff.”</i> "
        "Through 2021 the data look like a plateau. The productivity pick-up since 2022 is the first evidence that could "
        "point toward takeoff, and it is the result to watch.",
        "<i>“Computers complement humans.”</i> The cheapening of computers and software, together with the shift of "
        "investment into intellectual property, is the channel through which that complementarity would raise productivity. "
        "H8 tests whether it has.",
    ])
    st += [Paragraph("Limits", H2s)] + bullets([
        "Relative prices measure cost, not value. BLS quality adjustment for TVs and computers is large and debated.",
        "Services such as tuition, medical care and rent rise in relative price partly through Baumol's cost disease, not "
        "only because innovation stalled.",
        "Tuition here is the published price; the net price students pay after aid has risen less.",
        "Life expectancy depends on far more than medical technology; the post-2010 stall reflects drug overdoses and "
        "other causes as well as the pace of medical progress.",
        "Penn World Table productivity ends in 2023.",
    ])
    return st


# ---------------------------------------------------------------- H7
def section_h7(s):
    import math
    bs = s["by_sector"]; sw, ph = bs["Software and internet"], bs["Pharma, biotech, medical"]
    big = s["software_1bn"]
    yr = s["software_median_by_year"]
    G = {(r["sample"], r["x"]): r for r in s["growth_regressions"]}
    V = {(r["sample"], r["x"]): r for r in s["valuation_regressions"]}
    eff = {int(k): v for k, v in s["efficiency"].items()}
    pct = lambda c: 100 * (math.exp(c) - 1)
    g = lambda smp, x: 0.1 * G[(smp, x)]["coef"]                     # log points of growth per +10 pts of revenue spent
    last = max(eff)
    pre_eff = sum(eff[y]["median"] for y in range(2012, 2015)) / 3
    mid_eff = sum(eff[y]["median"] for y in range(2015, 2021)) / 6
    late_eff = sum(eff[y]["median"] for y in range(2023, last + 1)) / (last - 2022)
    st = [Paragraph("H7 · “Sales matters just as much as product” (ch. 2 and 11)", H1s),
          Paragraph("<b>Claim.</b> “Focus on product, not sales” was one of the false lessons of the dot-com crash; in "
                    "fact “sales matters just as much as product”. Engineers underrate distribution, and a business works "
                    "only when the lifetime value of a customer exceeds the cost of acquiring one.", P),
          Paragraph("<b>Verdict: holds, with a caveat about investors.</b> Technology firms spend more on selling than on "
                    "building, and sales spending is followed by faster growth than R&amp;D spending is. Investors, though, "
                    "pay more for R&amp;D, and selling has become less efficient since 2022.", P),
          Paragraph("What the data show", H2s)] + bullets([
        f"<b>Tech firms spend more on selling than on building.</b> Among {s['firms_both']:,} firms that report both lines "
        f"({s['n_both']:,} firm-years, 2011–2025), the median software and internet firm spends {sw['median_sm_pct']:.0f}% of "
        f"revenue on sales and marketing and {sw['median_rd_pct']:.0f}% on R&amp;D; sales spending is the larger of the two in "
        f"{sw['pct_firms_sm_above_rd']:.0f}% of firm-years. The pattern holds for the {big['firms']} software firms with "
        f"$1 billion or more in revenue ({big['median_sm_pct']:.0f}% against {big['median_rd_pct']:.0f}%). Pharma, biotech and "
        f"medical firms that report sales separately spend {ph['median_sm_pct']:.0f}% on sales against "
        f"{ph['median_rd_pct']:.0f}% on R&amp;D.",
        f"<b>Since the book, software has tilted back toward product.</b> The median software firm's sales spending peaked at "
        f"{yr['2014']['sm_int']:.0f}% of revenue in 2014 and fell to {yr[str(last)]['sm_int']:.0f}% in {last}; its R&amp;D "
        f"rose from {yr['2011']['rd_int']:.0f}% in 2011 to {yr[str(last)]['rd_int']:.0f}%. Sales still outspends R&amp;D, "
        f"but the gap halved.",
        f"<b>Sales spending is followed by growth.</b> Comparing firms in the same industry and year, spending 10 more points "
        f"of revenue on sales and marketing goes with {pct(g('all', 'sm_int')):.1f}% faster revenue growth the next year; "
        f"the same extra spending on R&amp;D goes with {pct(g('all', 'rd_int')):.1f}%. For software firms alone the figures "
        f"are {pct(g('software and internet', 'sm_int')):.1f}% and {pct(g('software and internet', 'rd_int')):.1f}% (the "
        f"R&amp;D figure is not statistically different from zero). Before the book the sales effect was "
        f"{pct(g('all, 2011-2014', 'sm_int')):.1f}%; since, {pct(g('all, 2015+', 'sm_int')):.1f}%.",
        f"<b>Investors pay more for product.</b> Holding growth and margins fixed, 10 more points of revenue spent on "
        f"R&amp;D goes with a {pct(V[('all', 'rd_int')]['coef'] / 10):.0f}% higher value relative to revenue, against "
        f"{pct(V[('all', 'sm_int')]['coef'] / 10):.0f}% for sales and marketing (software: "
        f"{pct(V[('software and internet', 'rd_int')]['coef'] / 10):.0f}% and {pct(V[('software and internet', 'sm_int')]['coef'] / 10):.0f}%). "
        f"The market treats R&amp;D as building a lasting asset and sales spending as a running cost.",
        f"<b>Selling has become harder.</b> For each dollar of the previous year's sales spending, the median software firm "
        f"added {pre_eff:.2f} dollars of new revenue in 2012–14 and {mid_eff:.2f} in 2015–20. After a peak of "
        f"{eff[2021]['median']:.2f} in 2021, it fell to {late_eff:.2f} in 2023–{last}, the lowest in the series. Large firms "
        f"held up better ({eff[last]['median_rev_1bn_plus']:.2f} in {last}).",
    ])
    st += fig("h7_fig1_intensity", "Figure H7-1. Median sales-and-marketing and R&amp;D spending as a share of revenue, "
              "firms reporting both, 2011–2025.")
    st += fig("h7_fig3_coefficients", "Figure H7-2. Left: next-year revenue growth associated with spending 10 more points "
              "of revenue on each. Right: value relative to revenue. Same industry and year compared; 95% intervals.")
    st += fig("h7_fig2_efficiency", "Figure H7-3. New revenue per dollar of the previous year's sales and marketing, median "
              "software and internet firm.", width=5.8)
    st += [Paragraph("How this compares with the book notes", H2s)] + bullets([
        "<i>“Sales is often ignored, but it is important.”</i> Supported. For software firms sales is the largest single "
        "expense after the cost of the product itself, and the spending is followed by growth.",
        "<i>“Effective distribution involves a higher customer lifetime value and a lower customer acquisition cost.”</i> The "
        "falling sales efficiency since 2022 means acquiring a dollar of revenue now costs more than it did, which squeezes "
        "exactly this margin.",
        "<i>“Building a product is just one side of the ledger.”</i> The valuation result adds a twist: investors still give "
        "more credit for the product side, perhaps because R&amp;D builds the proprietary technology Thiel lists first among "
        "the sources of monopoly.",
    ])
    st += [Paragraph("Limits", H2s)] + bullets([
        "Only firms reporting sales and marketing as a separate line are included (about 1,000 firms), mostly in software, "
        "healthcare and consumer goods.",
        "Spending is chosen, not assigned: firms with good growth prospects may spend more on selling because it pays them "
        "to, so these are associations, not effects.",
        "R&amp;D is expensed under US accounting, which makes R&amp;D-heavy firms look less profitable and may inflate their "
        "value-to-revenue ratio.",
    ])
    return st


# ---------------------------------------------------------------- H8
def section_h8(s):
    E, L = s["end_year"], s["last_aggregate_year"]
    ag = s["aggregate_periods"]; lv = s["employment_levels"]
    late, b1 = ag[f"2014-{L}"], ag["1995-2004"]
    qc, qr, qa, qp = s["quintiles_computer"], s["quintiles_repetitive"], s["quintiles_automation"], s["quintiles_pay"]
    lo, hi = "Q1 (least)", "Q5 (most)"
    R = {(r["y"], r["spec"].count("+"), r["x"]): r for r in s["regressions"]}
    A = {(r["y"], r["period"], r["spec"].count("+"), r["x"]): r for r in s["ai"]["regressions"]}
    mv = s["movers"]; dec = [m for m in mv if m["list"] == "fastest decline"]; grw = [m for m in mv if m["list"] == "fastest growth"]
    nm = lambda m: m["title"].replace(" (and related)", "").split(",")[0].lower()
    ex = {m["title"].split(" (")[0]: m for m in s["ai"]["most_exposed"]}
    aq_pre, aq_post = s["ai"]["quintiles_pre"], s["ai"]["quintiles_post"]
    ppt = lambda r, unit="": f"{r['coef']:+.2f} points{unit}, p = {r['p']:.2f}"
    st = [Paragraph("H8 · “Computers complement humans” (ch. 12)", H1s),
          Paragraph("<b>Claim.</b> Computers and people are good at different things, so computers make people more "
                    "productive rather than replacing them. Fears of technological unemployment are misplaced, and the "
                    "best businesses will “empower people rather than make them obsolete”.", P),
          Paragraph("<b>Verdict: mostly holds.</b> There was no technological unemployment: since 2014 productivity, pay "
                    "and employment all rose, and jobs that use computers most grew fastest. But computers did replace "
                    "people in routine clerical work, and the gains in computer-heavy jobs largely reflect the rising "
                    "demand for skilled, well-paid work. So far AI exposure shows no loss of jobs.", P),
          Paragraph("What the data show", H2s)] + bullets([
        f"<b>No technological unemployment.</b> From 2014 to {L} output per hour grew {late['Labour productivity, % per year']:.1f}% "
        f"a year and real median weekly earnings {late['Real median weekly earnings, % per year']:.1f}% a year, the fastest pay "
        f"growth of the four periods since 1979 (the IT boom of 1995–2004 managed {b1['Real median weekly earnings, % per year']:.1f}%). "
        f"The employment rate of people aged 25–54 rose from {lv['2014']['prime_age_employment_rate']:.1f}% to "
        f"{lv[str(L)]['prime_age_employment_rate']:.1f}%, close to its 2000 peak of {lv['2000']['prime_age_employment_rate']:.1f}%. "
        f"The one cloud is the labour share of income, which fell another {abs(late['Labour share, change in points']):.1f} points.",
        f"<b>Jobs that use computers most grew fastest.</b> Ranking {s['coverage']['groups']} occupations by how important "
        f"computers are to the work (O*NET) and splitting them into fifths of employment, the most computer-intensive fifth "
        f"grew {qc[hi]['emp_growth_pct_yr']:.1f}% a year from 2014 to {E}, against {qc[lo]['emp_growth_pct_yr']:.1f}% for the "
        f"least. The most repetitive fifth of jobs shrank ({qr[hi]['emp_growth_pct_yr']:+.1f}% a year) while the least "
        f"repetitive grew {qr[lo]['emp_growth_pct_yr']:.1f}%. How automated a job already is made no clear difference.",
        f"<b>But computers did replace some people.</b> The fastest-shrinking large occupations are mostly computer-using "
        f"clerical and routine jobs: {', '.join(nm(m) for m in dec[:3])}, {nm(dec[4])}, {nm(dec[6])} and data entry keyers, "
        f"most of them losing more than half their workers. Computer programmers fell from "
        f"{dec[1]['emp_2014'] / 1e3:.0f},000 to {dec[1]['emp_2025'] / 1e3:.0f},000 while the software developer group grew "
        f"from {grw[-2]['emp_2014'] / 1e6:.1f} to {grw[-2]['emp_2025'] / 1e6:.1f} million: within computing, the narrower coding "
        f"job shrank (partly through reclassification in 2018) while the broader design job grew.",
        f"<b>Computer use partly stands in for skill.</b> Computer-heavy jobs pay more (correlation with 2014 pay "
        f"{s['corr_computer_logpay']:.2f}), and the best-paid fifth of jobs grew {qp[hi]['emp_growth_pct_yr']:.1f}% a year. "
        f"Comparing jobs with the same starting pay, more computer use goes with slightly slower employment growth "
        f"({ppt(R[('g_emp', 3, 'z_computer')], ' a year per standard deviation')}). What grew is skilled work that uses computers, "
        f"which fits Thiel's complementarity but does not show computers alone creating jobs.",
        f"<b>Pay grew fastest at the bottom.</b> Real wages rose {qp[lo]['real_wage_growth_pct_yr']:.1f}% a year in the "
        f"lowest-paid fifth of jobs and {qp[hi]['real_wage_growth_pct_yr']:.1f}% in the highest, so computer-heavy jobs saw "
        f"slower pay growth ({qc[hi]['real_wage_growth_pct_yr']:.1f}% against {qc[lo]['real_wage_growth_pct_yr']:.1f}%). This fits "
        f"the widely reported squeeze of pay differences after 2020 rather than computers lowering pay.",
        f"<b>AI: no job losses yet, slower pay.</b> The fifth of jobs most exposed to AI grew "
        f"{aq_post[hi]['emp_growth_pct_yr']:.1f}% a year from 2022 to {E}, faster than before ChatGPT "
        f"({aq_pre[hi]['emp_growth_pct_yr']:.1f}%) and faster than the least exposed ({aq_post[lo]['emp_growth_pct_yr']:.1f}%). "
        f"With computer use and pay held fixed, AI exposure has no significant link to employment after 2022 "
        f"({ppt(A[('g_emp', 'post', 2, 'z_aioe')], ' a year per standard deviation')}) but goes with slower real wage growth "
        f"({ppt(A[('g_wage', 'post', 2, 'z_aioe')])}). "
        f"Within exposed fields the split repeats: loan interviewers and clerks fell "
        f"{abs(ex['Loan Interviewers and Clerks']['g_emp_post']):.0f}% a year and loan officers "
        f"{abs(ex['Loan Officers']['g_emp_post']):.0f}%, while financial managers grew {ex['Financial Managers']['g_emp_post']:.0f}%.",
    ])
    st += fig("h8_fig1_aggregate", f"Figure H8-1. Left: output per hour and real median weekly earnings, 1979 = 100. Right: "
              f"employment rate of people aged 25–54 and the unemployment rate, annual averages to {L}.")
    st += fig("h8_fig2_quintiles", f"Figure H8-2. Employment and real wage growth, 2014–{E}, by fifths of 2014 employment "
              "ranked on three O*NET job traits.")
    st += fig("h8_fig3_scatter", f"Figure H8-3. Employment growth 2014–{E} against the importance of computers to the job, "
              "occupations with 20,000 or more workers in 2014.", width=5.6)
    st += fig("h8_fig4_ai", f"Figure H8-4. Employment and real wage growth by AI exposure, before (2014–2022) and after "
              f"(2022–{E}) ChatGPT.")
    st += [Paragraph("How this compares with the book notes", H2s)] + bullets([
        "<i>“Computers are complements for humans, not substitutes for them.”</i> True for the economy as a whole and for "
        "skilled work; false for routine clerical work, where computers did the substituting.",
        "<i>“Fears of technological unemployment are misplaced.”</i> Supported. A decade of rising productivity ended with the "
        "prime-age employment rate near its 2000 peak and real pay growing faster than in the earlier periods since 1979.",
        "<i>“The most valuable businesses will empower people rather than make them obsolete.”</i> Not tested directly. The "
        "early AI evidence fits it so far, but three years is short, and pay in exposed jobs is already lagging.",
    ])
    from reportlab.platypus import CondPageBreak
    st += [CondPageBreak(1.6 * inch), Paragraph("Limits", H2s)] + bullets([
        f"Occupation codes changed in 2018; jobs are merged into {s['coverage']['groups']} stable groups covering "
        f"{s['coverage']['share_2014_emp_matched']:.0f}% of employment. A few groups (software developers, for one) combine "
        "several occupations.",
        "O*NET describes jobs as they are now. Occupations that computers had already shrunk before 2014 do not show up.",
        "Employment shifts also reflect demand, trade and an ageing population (home health aides more than doubled, "
        "to 4.3 million). These are associations, not effects.",
        "The AI exposure index (Felten, Raj and Seamans 2021) predates generative AI and measures overlap with AI abilities, "
        "not actual use.",
    ])
    return st


VERDICTS = {1: "Partly holds", 2: "Partly holds", 3: "Holds, with a twist", 4: "Mostly holds", 5: "Holds", 6: "Mostly holds", 7: "Holds, with a caveat", 8: "Mostly holds"}
SECTIONS = {1: section_h1, 2: section_h2, 3: section_h3, 4: section_h4, 5: section_h5, 6: section_h6, 7: section_h7, 8: section_h8}


# ---------------------------------------------------------------- report
def build():
    s1 = load(1)
    doc = SimpleDocTemplate(str(OUT / "Zero_to_One_Findings.pdf"), pagesize=letter, leftMargin=0.85 * inch,
                            rightMargin=0.85 * inch, topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                            title="Testing Zero to One: findings", author="Sarthak Pattnaik")
    st = [Paragraph("Testing <i>Zero to One</i>", T),
          Paragraph("A claim-by-claim test of Peter Thiel's arguments against data from before and after the book's "
                    "publication (16 September 2014). Findings log, updated as each notebook runs.", SUB),
          Paragraph("How to read this", H2s),
          Paragraph("Each section takes one specific claim, states what would count for and against it, and reports what "
                    "the data show. Verdicts use four grades: <b>holds</b>, <b>partly holds</b>, <b>does not hold</b>, and "
                    "<b>untestable</b> with public data. Numbers are recomputed from the notebooks' outputs each time this "
                    "report is built.", P),
          Paragraph("Scorecard", H2s)]
    rows = [["#", "Claim (chapter)", "Verdict", "Status"]]
    for n, c in [(1, "Power law: a few winners produce most of the value (7)"),
                 (2, "Competition destroys profits; monopolies capture value (3–4)"),
                 (3, "Durability and last-mover advantage (5)"), (4, "Scorecard of the companies Thiel named (3, 5, 13)"),
                 (5, "Indefinite optimism produces companies without plans (2, 6, 13)"),
                 (6, "Stagnation: progress in bits, not atoms (1, conclusion)"), (7, "Sales matters as much as product (11)"),
                 (8, "Computers complement humans (12)"), (9, "Founders and organisation (9–10)")]:
        done = load(n) is not None and n in SECTIONS
        status = "Pending" if load(n) is None else ("Final" if done else "Results in, write-up pending")
        if n == 1 and H1_PROVISIONAL and done:
            status = "Provisional"
        rows.append([f"H{n}", c, VERDICTS.get(n, "—") if done else "—", status])
    st += [table(rows, [0.4, 3.9, 1.1, 1.3]), PageBreak()]
    first = True
    for n, fn in SECTIONS.items():
        sn = load(n)
        if sn:
            if not first:
                st.append(PageBreak())
            st += fn(sn); first = False
    doc.build(st, onLaterPages=lambda c, d: (c.setFont(BASE, 8), c.setFillColor(MUTED),
                                             c.drawRightString(7.65 * inch, 0.5 * inch, f"Testing Zero to One · page {d.page}")))
    print("wrote", OUT / "Zero_to_One_Findings.pdf")


if __name__ == "__main__":
    build()
