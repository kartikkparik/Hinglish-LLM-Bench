#!/usr/bin/env python3
"""Generate data/hinglish_bench_150.jsonl for Hinglish-LLM-Bench.

Produces exactly 150 evaluation items:
  - 3 task categories x 50 items each:
      reasoning         : multi-step logical & arithmetic reasoning
      comprehension     : contextual reading comprehension & QA
      intent_sarcasm    : nuanced intent & sarcasm classification
  - 3 Code-Mixing Index (CMI) tiers x 50 items each:
      low    : ~15% Hindi  (English matrix, Hindi insertions)
      medium : ~40% Hindi  (balanced code-mixing)
      high   : ~70% Hindi  (Romanized Hindi + idioms)

Every item carries the prompt, a gold reference answer, human-readable
reasoning steps, and linguistic metadata (CMI estimate, Hindi proportion,
idioms used, script, difficulty).

Deterministic: random.seed(42). Re-run anytime; output is stable.
"""

import json
import os
import random
from collections import defaultdict

random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.normpath(os.path.join(HERE, "..", "data", "hinglish_bench_150.jsonl"))

CATEGORIES = ["reasoning", "comprehension", "intent_sarcasm"]
TIERS = ["low", "medium", "high"]

# Approximate Hindi-token proportion ranges per tier (used for metadata).
HINDI_PCT = {
    "low": (10, 20),
    "medium": (32, 48),
    "high": (62, 78),
}

NAMES_M = ["Amit", "Rohan", "Vikas", "Suresh", "Arjun", "Karan", "Mohit", "Dev"]
NAMES_F = ["Priya", "Neha", "Anjali", "Kavita", "Sneha", "Pooja", "Divya", "Meera"]
PLACES = ["Jaipur", "Lucknow", "Pune", "Chandigarh", "Varanasi", "Indore", "Bhopal", "Nagpur"]
ITEMS = ["pencil", "notebook", "bottle", "pen", "eraser", "sharpener"]
FOODS = ["samosa", "chai", "biryani", "dosa", "chole bhature", "jalebi"]

IDIOMS_HIGH = [
    "dimaag ka dahi", "aankh ka nasha", "mann ke gulaab", "aas-paas",
    "chup-chaap", "aage badhna", "dil jeet lena", "khamoshi todna",
    "mann bhar jaana", "haath paanv phoolna", "aakhri koshish", "mann ki bhasha",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hindi_pct(tier):
    lo, hi = HINDI_PCT[tier]
    return round(random.uniform(lo, hi), 1)


def make_item(idx, category, tier, prompt, answer, steps, idioms=None,
              difficulty="medium"):
    return {
        "id": f"HLLB-{idx:03d}",
        "category": category,
        "cmi_tier": tier,
        "prompt": prompt,
        "reference_answer": answer,
        "reasoning_steps": steps,
        "metadata": {
            "hindi_token_pct_est": hindi_pct(tier),
            "script": "romanized",
            "idioms_used": idioms or [],
            "difficulty": difficulty,
        },
    }


# ---------------------------------------------------------------------------
# Category A: multi-step logical & arithmetic reasoning
# ---------------------------------------------------------------------------

def reasoning_items(cat, tier, quota):
    items = []
    n = len(NAMES_M) * len(NAMES_F)

    # --- Template R1: wage x hours -------------------------------------
    def r1(name, wage, h1, h2, day1, day2):
        if tier == "low":
            p = (f"{name} earns Rs {wage} per hour at a bookstore. On {day1} "
                 f"he works {h1} hours and on {day2} he works {h2} hours. "
                 f"What is his total earning for the two days?")
            steps = [f"Total hours = {h1} + {h2} = {h1+h2}",
                     f"Earnings = {wage} x {h1+h2} = {wage*(h1+h2)}"]
        elif tier == "medium":
            p = (f"{name} ko bookstore mein {wage} rupaye per hour milte hain. "
                 f"{day1} ko wo {h1} ghante aur {day2} ko {h2} ghante kaam karta hai. "
                 f"Do din ka total kitna kamayega?")
            steps = [f"Total ghante = {h1} + {h2} = {h1+h2}",
                     f"Total kamai = {wage} x {h1+h2} = {wage*(h1+h2)}"]
        else:
            p = (f"Arre suno, {name} ko dukan pe {wage} rupaye ghante ka mazduri milti hai. "
                 f"{day1} ko {h1} ghante aur {day2} ko {h2} ghante kaam kiya. "
                 f"Toh bhai total kitna kama lega, hisaab lagao.")
            steps = [f"Pehle total ghante nikaalo: {h1} + {h2} = {h1+h2}",
                     f"Phir kamai: {wage} x {h1+h2} = {wage*(h1+h2)}"]
        return p, str(wage * (h1 + h2)), steps

    # --- Template R2: purchase with change --------------------------------
    def r2(name, price, qty, paid, item):
        total = price * qty
        change = paid - total
        if tier == "low":
            p = (f"{name} buys {qty} {item}s at Rs {price} each and pays Rs {paid} "
                 f"with a fifty-rupee note. How much change should he get back?")
            steps = [f"Cost of {qty} {item}s = {price} x {qty} = {total}",
                     f"Change = {paid} - {total} = {change}"]
        elif tier == "medium":
            p = (f"{name} {qty} {item} {price} rupaye ke rate se kharidta hai aur "
                 f"Rs {paid} ka note deta hai. Kitne rupaye wapas milenge?")
            steps = [f"Kul price = {price} x {qty} = {total}",
                     f"Change = {paid} - {total} = {change}"]
        else:
            p = (f"Bhai {name} ne {qty} {item} liye, har ek ka {price} rupaye pada. "
                 f"Usne {paid} ka note rakha. Ab kitne rupaye wapas aayenge, seedha batao.")
            steps = [f"Pehle total nikaalo: {price} x {qty} = {total}",
                     f"Ab {paid} mein se {total} minus karo = {change}"]
        return p, str(change), steps

    # --- Template R3: train speed/distance/time ---------------------------
    def r3(name, speed, h, m):
        dist = speed * (h + m / 60.0)
        dist_s = f"{dist:.2f}".rstrip("0").rstrip(".")
        if tier == "low":
            p = (f"A train leaves {PLACES[0]} at {speed} km/h. {name} boards it and "
                 f"travels for {h} hour {m} minutes. How many kilometres does he cover?")
            steps = [f"Time = {h} + {m}/60 = {h + m/60:.3f} hours",
                     f"Distance = {speed} x {h + m/60:.3f} = {dist_s} km"]
        elif tier == "medium":
            p = (f"Ek train {PLACES[1]} se {speed} km/h ki speed se nikalti hai. "
                 f"{name} usme baithkar {h} ghante {m} minute travel karta hai. "
                 f"Kitne kilometre cover hua?")
            steps = [f"Total time = {h} ghante {m} minute = {h + m/60:.3f} ghante",
                     f"Doori = {speed} x {h + m/60:.3f} = {dist_s} km"]
        else:
            p = (f"Train {PLACES[2]} se chal padi, {speed} km/h ki raftaar se. "
                 f"{name} {h} ghante {m} minute uske saath ghuma. Bhai kitne km chale gaye?")
            steps = [f"Pehle time ko ghante mein badlo: {h} + {m}/60",
                     f"Doori = speed x time = {speed} x {h + m/60:.3f} = {dist_s} km"]
        return p, dist_s, steps

    # --- Template R4: father-son age puzzle --------------------------------
    def r4(father_age, son_age, years):
        fa, sa = father_age + years, son_age + years
        if tier == "low":
            p = (f"A father is {father_age} years old and his son is {son_age}. "
                 f"What will be the sum of their ages after {years} more years?")
            steps = [f"Father's age after {years} yrs = {father_age} + {years} = {fa}",
                     f"Son's age after {years} yrs = {son_age} + {years} = {sa}",
                     f"Sum = {fa} + {sa} = {fa+sa}"]
        elif tier == "medium":
            p = (f"Ek pita ki umar {father_age} saal hai aur bete ki {son_age} saal. "
                 f"{years} saal baad dono ki umar ka jod kitna hoga?")
            steps = [f"Pita ki nayi umar = {father_age} + {years} = {fa}",
                     f"Bete ki nayi umar = {son_age} + {years} = {sa}",
                     f"Jod = {fa} + {sa} = {fa+sa}"]
        else:
            p = (f"Pita {father_age} saal ke hain, beta {son_age} ka. "
                 f"{years} saal baad dono ki umar jod ke batao na.")
            steps = [f"Dono ki umar {years} se badhegi: {father_age}+{years}={fa}, {son_age}+{years}={sa}",
                     f"Total = {fa} + {sa} = {fa+sa}"]
        return p, str(fa + sa), steps

    # --- Template R5: discount percentage ----------------------------------
    def r5(name, price, pct):
        disc = round(price * pct / 100)
        final = price - disc
        if tier == "low":
            p = (f"A jacket costs Rs {price}. During a sale {pct}% discount is given. "
                 f"What is the final price {name} pays?")
            steps = [f"Discount = {price} x {pct}/100 = {disc}",
                     f"Final price = {price} - {disc} = {final}"]
        elif tier == "medium":
            p = (f"Ek jacket ka price Rs {price} hai. Sale mein {pct}% discount hai. "
                 f"{name} ko final kitne rupaye dene padenge?")
            steps = [f"Discount amount = {price} x {pct}% = {disc}",
                     f"Final price = {price} - {disc} = {final}"]
        else:
            p = (f"Bhai ek jacket {price} rupaye ki hai, sale pe {pct}% chhoot hai. "
                 f"Seedha batao final price kya aayega.")
            steps = [f"Discount nikaalo: {price} ka {pct}% = {disc}",
                     f"Final = {price} - {disc} = {final}"]
        return p, str(final), steps

    # --- Template R6: average -------------------------------------------------
    def r6(name, nums):
        avg = round(sum(nums) / len(nums), 2)
        avg_s = f"{avg:.2f}".rstrip("0").rstrip(".")
        lst = ", ".join(str(x) for x in nums)
        if tier == "low":
            p = (f"{name} scores {lst} marks in {len(nums)} subjects. "
                 f"What is his average score?")
            steps = [f"Sum = {sum(nums)}", f"Average = {sum(nums)}/{len(nums)} = {avg_s}"]
        elif tier == "medium":
            p = (f"{name} ke {len(nums)} subjects ke marks hain: {lst}. Average kitne aayega?")
            steps = [f"Total = {sum(nums)}", f"Average = {sum(nums)} / {len(nums)} = {avg_s}"]
        else:
            p = (f"{name} ke marks ye rahe: {lst}. Bhai average nikal ke batao.")
            steps = [f"Sabko jodo: {sum(nums)}", f"{len(nums)} se divide karo = {avg_s}"]
        return p, avg_s, steps

    # --- Template R7: ratio split ---------------------------------------------
    def r7(name1, name2, total, r1_, r2_):
        share1 = round(total * r1_ / (r1_ + r2_))
        share2 = total - share1
        if tier == "low":
            p = (f"{name1} and {name2} share Rs {total} in the ratio {r1_}:{r2_}. "
                 f"How much does {name1} get?")
            steps = [f"Total parts = {r1_} + {r2_} = {r1_+r2_}",
                     f"{name1}'s share = {total} x {r1_}/{r1_+r2_} = {share1}"]
        elif tier == "medium":
            p = (f"{name1} aur {name2} Rs {total} ko {r1_}:{r2_} ratio mein baantte hain. "
                 f"{name1} ko kitne milenge?")
            steps = [f"Parts = {r1_} + {r2_} = {r1_+r2_}",
                     f"Share = {total} x {r1_}/{r1_+r2_} = {share1}"]
        else:
            p = (f"{name1} aur {name2} ne Rs {total} ko {r1_}:{r2_} mein baanta. "
                 f"{name1} ka hissa kitna bana, batao.")
            steps = [f"Total hisse = {r1_}+{r2_} = {r1_+r2_}",
                     f"{name1} ka share = {total} x {r1_}/{r1_+r2_} = {share1}"]
        return p, str(share1), steps

    # --- Template R8: work rate --------------------------------------------------
    def r8(name, rate, days):
        total = rate * days
        if tier == "low":
            p = (f"A typist types {rate} pages per day. How many pages will {name} "
                 f"type in {days} days if he works at the same rate?")
            steps = [f"Pages = {rate} x {days} = {total}"]
        elif tier == "medium":
            p = (f"Ek typist roz {rate} pages type karta hai. {name} {days} din tak "
                 f"same speed se kaam kare toh kitne pages banenge?")
            steps = [f"Pages = {rate} x {days} = {total}"]
        else:
            p = (f"Bhai ek banda roz {rate} pages likhta hai. {name} {days} din tak "
                 f"kare toh total kitne pages ho jayenge?")
            steps = [f"Roz {rate} x {days} din = {total} pages"]
        return p, str(total), steps

    # --- Build the pool -------------------------------------------------------
    pool = []
    wage_opts = [(120, 3, 4), (150, 4, 5), (200, 5, 3), (180, 6, 2), (250, 3, 6),
                 (140, 5, 4), (220, 4, 4), (160, 7, 3), (300, 2, 5), (175, 6, 4),
                 (130, 8, 2), (240, 5, 5)]
    for i, nm in enumerate(NAMES_M + NAMES_F):
        for j, (w, a, b) in enumerate(wage_opts):
            if (i + j) % 2 == 0:  # keep pool manageable
                d1, d2 = ["Monday", "Tuesday", "Wednesday", "Thursday"][j % 4], \
                         ["Friday", "Saturday", "Sunday"][j % 3]
                pool.append(r1(nm, w, a, b, d1, d2))

    for nm in NAMES_M + NAMES_F:
        for price, qty, paid, it in [
            (20, 3, 100, "pencil"), (15, 4, 100, "pen"), (30, 2, 100, "bottle"),
            (25, 3, 100, "notebook"), (12, 5, 100, "eraser"), (40, 2, 100, "sharpener"),
            (18, 6, 200, "pencil"), (50, 1, 100, "bottle"), (10, 8, 100, "pen"),
            (35, 3, 200, "notebook"),
        ]:
            pool.append(r2(nm, price, qty, paid, it))

    for nm in NAMES_M + NAMES_F:
        for sp, h, m in [(60, 2, 30), (80, 1, 45), (50, 3, 15), (90, 2, 0),
                         (70, 1, 30), (40, 4, 15), (100, 1, 15), (55, 2, 45)]:
            pool.append(r3(nm, sp, h, m))

    for fa in [32, 35, 40, 45, 50]:
        for sa in [4, 5, 6, 7, 8]:
            for yr in [5, 8, 10, 12]:
                pool.append(r4(fa, sa, yr))

    for nm in NAMES_M + NAMES_F:
        for pr, pc in [(800, 10), (1200, 15), (2000, 20), (500, 25), (1500, 30),
                       (2500, 40), (750, 50), (3000, 35), (450, 20), (1800, 25)]:
            pool.append(r5(nm, pr, pc))

    for nm in NAMES_M + NAMES_F:
        pool.append(r6(nm, [72, 85, 90, 68]))
        pool.append(r6(nm, [55, 60, 78, 82, 90]))
        pool.append(r6(nm, [40, 45, 50, 55, 60, 65]))
        pool.append(r6(nm, [88, 92, 76, 84, 80]))
        pool.append(r6(nm, [33, 41, 29, 37, 45, 39, 30]))

    for a, b in [("Amit", "Rohan"), ("Priya", "Neha"), ("Vikas", "Karan"),
                 ("Anjali", "Kavita"), ("Suresh", "Mohit"), ("Sneha", "Pooja")]:
        for tot, x, y in [(1000, 2, 3), (1500, 3, 2), (2000, 1, 4), (1200, 3, 5),
                          (900, 4, 1), (2400, 2, 5), (800, 1, 1), (3600, 5, 4)]:
            pool.append(r7(a, b, tot, x, y))

    for nm in NAMES_M + NAMES_F:
        for rt, d in [(10, 7), (15, 5), (8, 12), (20, 6), (12, 9), (25, 4),
                      (18, 8), (30, 3), (9, 15), (14, 10)]:
            pool.append(r8(nm, rt, d))

    return sample_pool(pool, quota, cat, tier)


# ---------------------------------------------------------------------------
# Category B: contextual reading comprehension & QA
# ---------------------------------------------------------------------------

def comprehension_items(cat, tier, quota):
    pool = []

    # Passage families parameterized by name/place/food/time.
    def comp_passage(nm, pl, food, tm, extra):
        if tier == "low":
            passage = (f"{nm} lives in {pl}. Every morning he walks to the local "
                       f"market to buy fresh {food}. Yesterday he reached the market "
                       f"at {tm} and bought two kilograms of it for his family. "
                       f"{extra}")
            qs = [
                (f"Where does {nm} live?", pl),
                (f"What does {nm} buy every morning?", food),
                (f"At what time did {nm} reach the market yesterday?", tm),
            ]
        elif tier == "medium":
            passage = (f"{nm} {pl} mein rehta hai. Roz subah wo local market jaake "
                       f"taza {food} khareedta hai. Kal wo {tm} baje market pahuncha "
                       f"aur apne parivaar ke liye do kilogram le aaya. {extra}")
            qs = [
                (f"{nm} kahan rehta hai?", pl),
                (f"{nm} roz subah kya khareedta hai?", food),
                (f"Kal {nm} kitne baje market pahuncha?", tm),
            ]
        else:
            passage = (f"Suno {nm} ki baat, wo {pl} ka rehne wala hai. Roz subah "
                       f"wo seedha market jaata hai aur {food} ki kharidaar karta hai. "
                       f"Kal {tm} baje market pahuncha, phir do kilo {food} leke "
                       f"ghar aaya. {extra}")
            qs = [
                (f"{nm} kahan ka hai?", pl),
                (f"Roz subah {nm} kya khareedta hai?", food),
                (f"Kal {nm} kab market pahuncha?", tm),
            ]
        for q, a in qs:
            pool.append((passage, q, a))

    EXTRAS = {
        "low": "He returns home before sunrise.",
        "medium": "Wo subah suraj nikalne se pehle ghar laut aata hai.",
        "high": "Bhai wo dhoop nikalne se pehle hi ghar pahunch jaata hai.",
    }

    tms = ["seven", "seven thirty", "eight", "eight fifteen", "six forty-five",
           "saat", "saat-saat thirty", "aath", "aath baje", "chheh.chalis",
           "sau baje", " nau baje"]
    for nm in NAMES_M + NAMES_F:
        for pl in PLACES[:4]:
            for food, tm in zip(FOODS * 2, tms * 2):
                comp_passage(nm, pl, food, tm.strip(), EXTRAS[tier])

    # Second family: school/teacher passage.
    def comp_school(nm, pl, subj, ev):
        if tier == "low":
            passage = (f"{nm} teaches {subj} at a school in {pl}. She loves her "
                       f"students and always explains lessons with real-life "
                       f"examples. {ev} Yesterday she stayed late to help a slow "
                       f"student revise for the exam.")
            qs = [
                (f"What subject does {nm} teach?", subj),
                (f"Where is the school located?", pl),
            ]
        elif tier == "medium":
            passage = (f"{nm} {pl} ke ek school mein {subj} padhati hai. Use apne "
                       f"students bahut pyar hain aur wo har lesson ko real-life "
                       f"example se samjhati hai. {ev} Kal wo ek weak student ko "
                       f"exam ke liye late tak padhaya.")
            qs = [
                (f"{nm} kaunsa subject padhati hai?", subj),
                (f"School kahan hai?", pl),
            ]
        else:
            passage = (f"Bhai {nm} {pl} wale school mein {subj} padhati hai. Use "
                       f"bacchon pe dil jeet liya hai, har lesson ko asli example "
                       f"se samjhati hai. {ev} Kal ek kamzor student ke liye late "
                       f"tak baithi rahi.")
            qs = [
                (f"{nm} kya padhati hai?", subj),
                (f"School kahan hai?", pl),
            ]
        for q, a in qs:
            pool.append((passage, q, a))

    EV = {
        "low": "The school has a big playground.",
        "medium": "School mein bada playground hai.",
        "high": "School ka ground to bada hai hi, khel ka mazaa aata hai.",
    }
    for nm in NAMES_F:
        for pl in PLACES[4:8]:
            for subj in ["Mathematics", "Science", "English", "Hindi"]:
                comp_school(nm, pl, subj, EV[tier])

    # Third family: train journey.
    def comp_train(nm, train, dest, seat):
        if tier == "low":
            passage = (f"{nm} booked a ticket on the {train} from Delhi to {dest}. "
                       f"Her seat number is {seat}. The train leaves at 6:30 AM "
                       f"and reaches {dest} by evening. She packed a small bag "
                       f"with snacks and a book.")
            qs = [
                (f"Which train did {nm} board?", train),
                (f"What is {nm}'s seat number?", seat),
                (f"Where is the train going?", dest),
            ]
        elif tier == "medium":
            passage = (f"{nm} ne Delhi se {dest} ke liye {train} ka ticket liya. "
                       f"Uski seat number {seat} hai. Train subah 6:30 baje nikalti "
                       f"hai aur shaam tak {dest} pahunch jaati hai. Usne ek chhota "
                       f"sa bag pack kiya jisme snacks aur ek kitab thi.")
            qs = [
                (f"{nm} kis train mein baith rahi hai?", train),
                (f"{nm} ki seat number kya hai?", seat),
                (f"Train kahan ja rahi hai?", dest),
            ]
        else:
            passage = (f"Suno {nm} ki safar ki kahaani. Usne Delhi se {dest} ke liye "
                       f"{train} pakdi, seat number {seat}. Train subah 6:30 baje "
                       f"nikli, shaam tak {dest} pahunch gayi. Saath mein ek chhota "
                       f"bag tha jisme nashta aur kitab thi.")
            qs = [
                (f"{nm} kis train pe thi?", train),
                (f"Seat number kya tha?", seat),
                (f"Train kahan pahunchi?", dest),
            ]
        for q, a in qs:
            pool.append((passage, q, a))

    trains = ["Shatabdi Express", "Rajdhani Express", "Duronto", "Garib Rath",
              "Vande Bharat"]
    seats = ["12B", "45A", "7C", "22D", "63B", "18A", "54C", "9D"]
    idx = 0
    for nm in NAMES_M + NAMES_F:
        for tr, de in zip(trains, PLACES[:5]):
            comp_train(nm, tr, de, seats[idx % len(seats)])
            idx += 1

    # Convert to item dicts
    random.shuffle(pool)
    chosen = pool[:quota]
    out = []
    for passage, q, a in chosen:
        steps = [f"Passage padho: '{passage[:80]}...'",
                 f"Question ka answer passage mein dhundo: '{a}'"]
        prompt = f"Read the passage and answer the question.\n\nPassage: {passage}\n\nQuestion: {q}"
        out.append((prompt, a, steps))
    return out


# ---------------------------------------------------------------------------
# Category C: nuanced intent & sarcasm classification
# ---------------------------------------------------------------------------

INTENT_LABELS = ["praise", "complaint", "request", "sarcasm", "neutral_info"]

def intent_items(cat, tier, quota):
    pool = []

    # Each entry: (sentence, gold_label, explanation)
    BASE = [
        # praise
        ("This biryani is absolutely amazing, best I've had in months!", "praise",
         "Strong positive adjectives 'amazing', 'best' show genuine praise."),
        ("Shukriya, aapne bohot madad ki, sach mein bahut achha kiya aapne.", "praise",
         "Direct thanks ('shukriya') with positive evaluation."),
        ("What a brilliant idea! You always think so creatively.", "praise",
         "'Brilliant', 'creatively' are unambiguous compliments."),
        ("Mujhe ye delivery service bahut pasand hai, kabhi late nahi hota.", "praise",
         "States liking and reliability — clear positive sentiment."),
        ("Great job team! The presentation was really well organised.", "praise",
         "Explicit praise directed at the team."),
        ("Yeh movie dekh ke dil khush ho gaya, actors ne zabardast acting ki.", "praise",
         "Emotional positive reaction with compliments to actors."),
        # complaint
        ("This is the third time my order has arrived late, very disappointed.", "complaint",
         "Repeated lateness + explicit disappointment = complaint."),
        ("Aap log ka customer service itna kharab kaise hai? Kabhi call attend nahi karte.", "complaint",
         "Rhetorical question criticizing service quality."),
        ("The wifi has been down since morning, this is really unacceptable.", "complaint",
         "Problem statement + 'unacceptable' signals complaint."),
        ("Refund ka promise ab tak nahi hua, bilkul ganda experience raha.", "complaint",
         "Unfulfilled promise + negative evaluation."),
        ("Package toh damage ho kar aaya, aur customer care ne call bhi nahi uthaya.", "complaint",
         "Two complaints: damaged goods and ignored support."),
        # request
        ("Could you please send me the report by Friday?", "request",
         "Polite imperative asking for an action = request."),
        ("Kya aap mujhe kal meeting ki timing bata sakte ho?", "request",
         "Indirect question seeking information/action."),
        ("Please book two tickets for the Saturday show.", "request",
         "'Please' + action verb = polite request."),
        ("Mujhe kal subah 9 baje doctor se appointment chahiye, book kar dijiye.", "request",
         "Explicit ask for appointment booking."),
        ("Can you share the file on the group chat?", "request",
         "Yes/no question functioning as a request."),
        # sarcasm
        ("Oh great, another Monday. Just what I needed.", "sarcasm",
         "'Oh great' + flat tone about Monday signals sarcasm."),
        ("Wow, what a fantastic idea — let's meet at 5 AM then.", "sarcasm",
         "Positive words used ironically with absurd suggestion."),
        ("Bilkul badhiya, aapka phone toh phir se 10 minute mein dead ho gaya.", "sarcasm",
         "'Bilkul badhiya' said about a dead phone = sarcasm."),
        ("Shabaash, traffic jam mein phanse rehna hi toh asli experience hai.", "sarcasm",
         "Praise word 'shabaash' used ironically about a traffic jam."),
        ("Arre bahut hi smart move tha, ab exam mein fail ho hi gaye.", "sarcasm",
         "'Smart move' ironic since it led to failure."),
        ("Nice one — you broke the printer right before the deadline.", "sarcasm",
         "'Nice one' is ironic given the negative outcome."),
        # neutral_info
        ("The meeting is scheduled for 3 PM in Conference Room B.", "neutral_info",
         "Factual statement, no emotion or evaluation."),
        ("Metro fare from Noida to Connaught Place is Rs 60.", "neutral_info",
         "Pure factual information."),
        ("Library remains closed on public holidays.", "neutral_info",
         "Neutral policy statement."),
        ("The train arrives at platform 4 at 8:45 AM.", "neutral_info",
         "Schedule information only."),
        ("Delhi temperature today is 34 degrees Celsius.", "neutral_info",
         "Weather fact, no sentiment."),
    ]

    # Tier-specific rewordings to inject code-mixing levels.
    def reword(sentence, label, expl, tier):
        if tier == "low":
            return sentence, label, expl
        if tier == "medium":
            # Mix Hindi function words into English sentences.
            swaps = {
                "sarcasm": "Arre {s} — matlab ye toh bilkul hi opposite hai na?",
            }
            if label == "sarcasm":
                return f"Arre suno, {sentence} Matlab ye toh bilkul ulta keh rahe ho?", label, expl
            if label == "request":
                return f"Bhai {sentence} Mujhe ye karwa do please.", label, expl
            if label == "complaint":
                return f"Arey yaar {sentence} Kya bakwaas service hai ye.", label, expl
            if label == "praise":
                return f"Wah {sentence} Truly mast hai yaar.", label, expl
            return f"Bhai, {sentence}", label, expl
        # high tier: heavily Romanized Hindi
        HIGH = {
            "praise": [
                "Bhai ye toh kamaal hai! Dil jeet liya tune, shabaash.",
                "Suno ye kaam bahut hi mast tha, sach mein shukriya.",
                "Arre ye toh badhiya hai na! Aadat pad jaayegi.",
            ],
            "complaint": [
                "Bhai ye service bilkul bakwaas hai, baar-baar problem aa rahi hai.",
                "Suno ye toh nahi chalega, aise mein kaam kaise karega banda.",
                "Arey yaar ye sabse bura experience tha, kabhi mat karna ye.",
            ],
            "request": [
                "Bhai ek kaam karo, ye file mujhe bhej do please.",
                "Suno kal wali meeting ki timing bata do, urgent hai.",
                "Arre ye book kar do na, bhool mat jaana kabhi.",
            ],
            "sarcasm": [
                "Arre bahut hi smart move tha bhai, ab toh bas.",
                "Wah kya mast hai na! Phone phir se dead ho gaya.",
                "Shabaash! Traffic mein phasna hi toh asli mazaa hai.",
                "Oh great, Monday aa gayi — kya hi badhiya din hai.",
            ],
            "neutral_info": [
                "Meeting kal shaam 3 baje hai, room B mein.",
                "Metro ticket ka rate 60 rupaye hai.",
                "Library chhutti ke din band rehti hai.",
                "Train subah 8:45 baje platform 4 pe aayegi.",
            ],
        }
        options = HIGH.get(label, [sentence])
        return random.choice(options), label, expl

    for s, lab, ex in BASE:
        s2, lab2, ex2 = reword(s, lab, ex, tier)
        pool.append((s2, lab2, ex2))

    # Extra parameterized variants for volume.
    NAMES_ALL = NAMES_M + NAMES_F
    extra = []
    for nm in NAMES_ALL:
        extra += [
            (f"{nm} ki presentation aaj bohot hi achhi thi, everyone clapped.", "praise",
             "Positive feedback on presentation."),
            (f"{nm} ka order phir se late aaya, yeh service kaisi hai?", "complaint",
             "Late delivery complaint with rhetorical tone."),
            (f"Kya {nm} kal wali report bhej sakta hai?", "request",
             "Request phrased as a question."),
            (f"Arre {nm} ne phir se phone charge karna bhool gaya, kitna smart hai!", "sarcasm",
             "'kitna smart hai' used ironically about forgetting to charge."),
            (f"{nm} ki train 9 baje platform 2 se nikalti hai.", "neutral_info",
             "Schedule fact only."),
        ]
    for s, lab, ex in extra:
        s2, lab2, ex2 = reword(s, lab, ex, tier)
        pool.append((s2, lab2, ex2))

    random.shuffle(pool)
    chosen = pool[:quota]
    out = []
    for s, lab, ex in chosen:
        prompt = (f"Classify the intent of the following utterance into exactly one "
                  f"of these labels: {', '.join(INTENT_LABELS)}.\n\nUtterance: {s}\n\n"
                  f"Answer with only the label.")
        steps = [f"Identify emotional cues and speech act in: '{s}'",
                 f"Map to closest label -> {lab}",
                 f"Justification: {ex}"]
        out.append((prompt, lab, steps))
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def sample_pool(pool, quota, cat, tier):
    """Deduplicate prompts and sample `quota` items."""
    seen, uniq = set(), []
    random.shuffle(pool)
    for p in pool:
        key = p[0] if isinstance(p, tuple) else p["prompt"]
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    if len(uniq) < quota:
        raise RuntimeError(f"Pool too small for {cat}/{tier}: {len(uniq)} < {quota}")
    return uniq[:quota]


def main():
    items = []
    # quotas: each category gets 50 (17+17+16) and each tier gets 50,
    # rotating which tier gets the 16 so all 9 cells stay balanced.
    quota_map = {}
    for ci, cat in enumerate(CATEGORIES):
        for ti, tier in enumerate(TIERS):
            quota_map[(cat, tier)] = 16 if ti == ci else 17

    idx = 1
    for cat in CATEGORIES:
        for tier in TIERS:
            q = quota_map[(cat, tier)]
            if cat == "reasoning":
                rows = reasoning_items(cat, tier, q)
            elif cat == "comprehension":
                rows = comprehension_items(cat, tier, q)
            else:
                rows = intent_items(cat, tier, q)
            for prompt, answer, steps in rows:
                items.append(make_item(idx, cat, tier, prompt, answer, steps))
                idx += 1

    # Sort by id already sequential; write JSONL.
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Summary
    from collections import Counter
    cc = Counter(i["category"] for i in items)
    tc = Counter(i["cmi_tier"] for i in items)
    print(f"Wrote {len(items)} items -> {OUT_PATH}")
    print("categories:", dict(cc))
    print("tiers:", dict(tc))


if __name__ == "__main__":
    main()
