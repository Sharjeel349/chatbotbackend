"""Robust Urdu script to Roman Urdu transliteration service.

Handles arbitrary Urdu queries, common academic terminology, conversational Urdu,
Whisper speech-to-text acoustic variations, and rule-based phonetic transliteration
for out-of-vocabulary words.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final


# Character Normalization map (Normalize Arabic/Persian variants to standard Urdu Unicode)
_CHAR_NORMALIZATION: Final[dict[str, str]] = {
    "\u064a": "\u06cc",  # Arabic Yeh (ي) -> Urdu Yeh (ی)
    "\u0649": "\u06cc",  # Alef Maksura (ى) -> Urdu Yeh (ی)
    "\u0643": "\u06a9",  # Arabic Kaf (ك) -> Urdu Keheh (ک)
    "\u06c0": "\u06c1",  # Heh with Yeh above (ۀ) -> Heh Gol (ہ)
    "\u0629": "\u06c1",  # Teh Marbuta (ة) -> Heh Gol (ہ)
    "\u0647": "\u06c1",  # Arabic Heh (ه) -> Urdu Heh Gol (ہ)
    "\u06be": "\u06be",  # Do-chashmi Heh (ھ)
    "\u0624": "\u0624",  # Waw with Hamza (ؤ)
    "\u0626": "\u0626",  # Yeh with Hamza (ئ)
    "\u0621": "\u0621",  # Isolated Hamza (ء)
    "\u06d2": "\u06d2",  # Bari Yeh (ے)
    "\u06ba": "\u06ba",  # Noon Ghunna (ں)
    "\u0679": "\u0679",  # Tteh (ٹ)
    "\u0688": "\u0688",  # Ddal (ڈ)
    "\u0691": "\u0691",  # Rreh (ڑ)
}

# Unicode Urdu script range for quick detection
_URDU_RANGE_PATTERN: Final[re.Pattern] = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")

# Comprehensive Vocabulary: Urdu Script -> Standard Roman Urdu
_URDU_TO_ROMAN_DICT: Final[dict[str, str]] = {
    # --- Academic & University Specific Terms (including Whisper mishearings) ---
    "یونیورسٹی": "university",
    "یونورسٹی": "university",
    "اینوستی": "university",
    "یونیورسٹیز": "universities",
    "فیس": "fees",
    "فیسیں": "fees",
    "فیسوں": "fees",
    "سبمٹ": "submit",
    "سبمیٹ": "submit",
    "سبب": "submit",
    "جمع": "jama",
    "امتحان": "exam",
    "امتحانات": "exams",
    "ایگزام": "exam",
    "ایگزامز": "exams",
    "اگزان": "exam",
    "اگزام": "exam",
    "پیپر": "paper",
    "پیپرز": "papers",
    "سی": "CGPA",
    "جی": "GPA",
    "پی": "P",
    "اے": "A",
    "سی جی پی اے": "CGPA",
    "جی پی اے": "GPA",
    "سیجaction": "CGPA",
    "سمسٹر": "semester",
    "سمسٹرز": "semesters",
    "کورس": "course",
    "کورسز": "courses",
    "مضمون": "subject",
    "مضامین": "subjects",
    "سبجیکٹ": "subject",
    "سبجیکٹس": "subjects",
    "گریڈ": "grade",
    "گریڈز": "grades",
    "نمبر": "marks",
    "مارکس": "marks",
    "پاس": "pass",
    "فیل": "fail",
    "پاسنگ": "passing",
    "فیلنگ": "failing",
    "ڈراپ": "drop",
    "فریز": "freeze",
    "انفریز": "unfreeze",
    "ایڈمیشن": "admission",
    "داخلہ": "admission",
    "داخلے": "admissions",
    "رجسٹریشن": "registration",
    "رجسٹرڈ": "registered",
    "اینرول": "enroll",
    "اینرولمنٹ": "enrollment",
    "داخل": "enrolled",
    "ڈگری": "degree",
    "ٹرانسکرپٹ": "transcript",
    "رزلٹ": "result",
    "نتیجہ": "result",
    "نتائج": "results",
    "پرفارمنس": "performance",
    "کارکردگی": "performance",
    "حاضری": "attendance",
    "اٹینڈنس": "attendance",
    "غیرحاضر": "absent",
    "غیرحاضری": "absence",
    "چالان": "challan",
    "قسط": "installment",
    "اقساط": "installments",
    "قسطوں": "installments",
    "سکالرشپ": "scholarship",
    "وظیفہ": "scholarship",
    "اسائنمنٹ": "assignment",
    "اسائنمنٹس": "assignments",
    "کوئز": "quiz",
    "کوئزز": "quizzes",
    "پروجیکٹ": "project",
    "پروجیکٹس": "projects",
    "لیب": "lab",
    "لیبز": "labs",
    "لیکچر": "lecture",
    "لیکچرز": "lectures",
    "شعبہ": "department",
    "ڈپارٹمنٹ": "department",
    "کلاس": "class",
    "کلاسز": "classes",
    "سیکشن": "section",
    "ٹیچر": "teacher",
    "ٹیچرز": "teachers",
    "استاد": "teacher",
    "اساتذہ": "teachers",
    "سر": "sir",
    "میم": "madam",
    "میڈم": "madam",
    "مشیر": "advisor",
    "ایڈوائزر": "advisor",
    "سپروائزر": "supervisor",
    "پروگرامنگ": "programming",
    "کوڈنگ": "coding",
    "کریڈٹ": "credit",
    "آرز": "hours",
    "گھنٹے": "ghantay",
    "کریڈٹ آورز": "credit hours",
    "پری ریکوئزٹ": "prerequisite",
    "پالیسی": "policy",
    "رول": "rule",
    "رولز": "rules",
    "قانون": "qanoon",
    "قوانین": "qawaneen",
    "شرائط": "sharaait",
    "نقل": "cheating",
    "چیٹنگ": "cheating",
    "پینلٹی": "penalty",
    "جرمانہ": "fine",

    # --- Pronouns & Demonstratives ---
    "میں": "main",
    "مجھ": "mujh",
    "مجھے": "mujhe",
    "میرا": "mera",
    "میری": "meri",
    "میرے": "mere",
    "ہم": "hum",
    "ہمیں": "humein",
    "ہمارا": "hamara",
    "ہماری": "hamari",
    "ہمارے": "hamare",
    "تو": "tu",
    "تجھے": "tujhe",
    "تیرا": "tera",
    "تیری": "teri",
    "تیرے": "tere",
    "تم": "tum",
    "تمہیں": "tumhein",
    "تمہارا": "tumhara",
    "تمہاری": "tumhari",
    "تمہارے": "tumhare",
    "آپ": "aap",
    "آپکو": "aap ko",
    "آپکا": "aap ka",
    "آپکی": "aap ki",
    "آپکے": "aap ke",
    "اپنا": "apna",
    "اپنی": "apni",
    "اپنے": "apne",
    "وہ": "woh",
    "اس": "us",
    "اسے": "usey",
    "اسکا": "uska",
    "اسکی": "uski",
    "اسکے": "uske",
    "ان": "un",
    "انہیں": "unhein",
    "انکا": "unka",
    "انکی": "unki",
    "انکے": "unke",
    "یہ": "yeh",
    "اس": "is",
    "اسے": "isey",
    "جس": "jis",
    "جن": "jin",
    "کس": "kis",
    "کن": "kin",
    "کسے": "kise",
    "کسی": "kisi",
    "کچھ": "kuch",
    "سب": "sab",
    "سارے": "saare",
    "ساری": "saari",
    "سارا": "saara",

    # --- Questions & Interrogatives ---
    "کیا": "kya",
    "کیوں": "kyun",
    "کب": "kab",
    "کہاں": "kahan",
    "کہیں": "kahin",
    "کیسا": "kaisa",
    "کیسی": "kaisi",
    "کیسے": "kaise",
    "کون": "kon",
    "کونسا": "konsa",
    "کونسی": "konsi",
    "کونسے": "konse",
    "کتنا": "kitna",
    "کتنی": "kitni",
    "کتنے": "kitne",
    "کدھر": "kidhar",

    # --- Common Verbs & Auxiliaries ---
    "ہے": "hai",
    "ہیں": "hain",
    "ہو": "ho",
    "ہوں": "hoon",
    "تھا": "tha",
    "تھی": "thi",
    "تھے": "the",
    "تھیں": "theen",
    "ہونا": "hona",
    "ہوتا": "hota",
    "ہوتی": "hoti",
    "ہوتے": "hote",
    "ہوئی": "hui",
    "ہوئے": "huye",
    "ہوا": "hua",
    "ہوگا": "hoga",
    "ہوگی": "hogi",
    "ہوں گے": "hongay",
    "ہونگے": "hongay",
    "ہوئیں": "hueen",
    "ہوئیں گی": "hoongi",
    "کرنا": "karna",
    "کرنے": "karne",
    "کرنی": "karni",
    "کرتا": "karta",
    "کرتی": "karti",
    "کرتے": "karte",
    "کریں": "karein",
    "کرے": "karey",
    "کیا": "kiya",
    "کی": "ki",
    "کیے": "kiye",
    "دینا": "dena",
    "دینے": "dene",
    "دینی": "deni",
    "دے": "de",
    "دی": "di",
    "دیا": "diya",
    "دیے": "diye",
    "دیں": "dein",
    "دیکھنا": "dekhna",
    "دیکھ": "dekh",
    "دیکھیں": "dekhein",
    "دیکھا": "dekha",
    "سکتا": "sakta",
    "سکتی": "sakti",
    "سکتے": "sakte",
    "سکوں": "sakoon",
    "سکیں": "sakein",
    "سکا": "saka",
    "سکی": "saki",
    "سکے": "sake",
    "دیستکتا": "de sakta",
    "دیسکتا": "de sakta",
    "کرسکتا": "kar sakta",
    "کرسکتی": "kar sakti",
    "کرسکتے": "kar sakte",
    "ہوسکتا": "ho sakta",
    "ہوسکتی": "ho sakti",
    "ہوسکتے": "ho sakte",
    "لینا": "lena",
    "لینے": "lene",
    "لینی": "leni",
    "لے": "le",
    "لی": "li",
    "لیا": "liya",
    "لیے": "liye",
    "لیں": "lein",
    "جانا": "jana",
    "جانے": "jane",
    "جاتا": "jata",
    "جاتی": "jati",
    "جاتے": "jate",
    "جا": "ja",
    "گیا": "gaya",
    "گئی": "gayi",
    "گئے": "gaye",
    "جائے": "jaye",
    "جائیں": "jayein",
    "آنا": "aana",
    "آنے": "aane",
    "آتا": "aata",
    "آتی": "aati",
    "آتے": "aate",
    "آیا": "aaya",
    "آئی": "aayi",
    "آئے": "aaye",
    "بتانا": "batana",
    "بتانے": "batane",
    "بتاو": "batao",
    "بتائیں": "batain",
    "بتادیں": "bata dein",
    "پوچھنا": "poochna",
    "پوچھ": "pooch",
    "پوچھیں": "poochein",
    "بولنا": "bolna",
    "بولو": "bolo",
    "بولیں": "bolein",
    "کہنا": "kehna",
    "کہو": "kaho",
    "کہیں": "kahein",
    "کہا": "kaha",
    "سمجھنا": "samajhna",
    "سمجھ": "samajh",
    "سمجھائیں": "samjhayein",
    "لگنا": "lagna",
    "لگتا": "lagta",
    "لگتی": "lagti",
    "لگتے": "lagte",
    "چاہتا": "chahta",
    "چاہتی": "chahti",
    "چاہتے": "chahte",
    "چاہیے": "chahiye",
    "پڑتا": "parta",
    "پڑتی": "parti",
    "پڑتے": "parte",
    "پڑا": "para",
    "پڑی": "pari",
    "پڑے": "pare",

    # --- Prepositions, Conjunctions & Particles ---
    "کا": "ka",
    "کی": "ki",
    "کے": "ke",
    "کو": "ko",
    "سے": "se",
    "میں": "mein",
    "پر": "par",
    "پہ": "pe",
    "تک": "tak",
    "اور": "aur",
    "یا": "ya",
    "بھی": "bhi",
    "اوی": "abhi",
    "تو": "to",
    "ہی": "hi",
    "نہ": "na",
    "نہیں": "nahi",
    "نھیں": "nahi",
    "نہیں": "nahi",
    "مگر": "magar",
    "لیکن": "lekin",
    "اگر": "agar",
    "چونکہ": "chunkay",
    "اسلئے": "is liye",
    "اسلیے": "is liye",
    "کیونکہ": "kyun ke",
    "پس": "pas",
    "ساتھ": "saath",
    "پاس": "paas",
    "سامنے": "samnay",
    "پیچھے": "peeche",
    "اوپر": "ooper",
    "نیچے": "neeche",
    "اندر": "andar",
    "باہر": "bahar",
    "درمیان": "darmiyan",
    "بارے": "baare",
    "طرف": "taraf",
    "بغیر": "baghair",
    "علاوہ": "ilawa",
    "سوائے": "siwaye",

    # --- Time, Adverbs & Qualifiers ---
    "اب": "ab",
    "ابھی": "abhi",
    "آج": "aaj",
    "کل": "kal",
    "پرسوں": "parson",
    "پہلے": "pehle",
    "بعد": "baad",
    "دوبارہ": "dobara",
    "ہمیشہ": "hamesha",
    "کبھی": "kabhi",
    "جلدی": "jaldi",
    "دیر": "dair",
    "وقت": "waqt",
    "سال": "saal",
    "مہینہ": "maheena",
    "مہینے": "maheenay",
    "ہفتہ": "hafta",
    "ہفتے": "haftay",
    "دن": "din",
    "رات": "raat",
    "صبح": "subah",
    "شام": "shaam",
    "بہت": "bohot",
    "زیادہ": "zyada",
    "تھوڑا": "thora",
    "تھوڑی": "thori",
    "تھوڑے": "thore",
    "کم": "kam",
    "صرف": "sirf",
    "بالکل": "bilkul",
    "تقریباً": "taqreeban",
    "تقریبا": "taqreeban",

    # --- Politeness, Greetings & Common Words ---
    "سلام": "Salam",
    "السلام": "Assalam",
    "علیکم": "Alaikum",
    "وعلیکم": "Walekum",
    "ہیلو": "Hello",
    "شکریہ": "shukriya",
    "مہربانی": "meherbani",
    "معذرت": "maazrat",
    "خوش": "khush",
    "آمدید": "aamdeed",
    "خدا": "Khuda",
    "اللہ": "Allah",
    "حافظ": "hafiz",
    "ٹھیک": "theek",
    "اچھا": "acha",
    "اچھی": "achi",
    "اچھے": "ache",
    "برا": "bura",
    "بری": "buri",
    "برے": "bure",
    "صحیح": "sahi",
    "غلط": "ghalat",
    "آسان": "aasan",
    "مشکل": "mushkil",
    "نام": "naam",
    "بات": "baat",
    "باتیں": "baatein",
    "سوال": "sawal",
    "سوالات": "sawalaat",
    "جواب": "jawab",
    "معلومات": "information",
    "تفصیل": "detail",
    "تفصیلات": "details",
    "مدد": "madad",
    "مسئلہ": "masla",
    "مسائل": "masayil",
    "پریشانی": "pareshani",

    # --- Numbers ---
    "صفر": "zero",
    "ایک": "aik",
    "دو": "do",
    "تین": "teen",
    "چار": "chaar",
    "پانچ": "paanch",
    "چھ": "chhay",
    "سات": "saat",
    "آٹھ": "aath",
    "نو": "nau",
    "دس": "das",
    "گیارہ": "gyarah",
    "بارہ": "barah",
    "تیرہ": "terah",
    "چودہ": "chaudah",
    "پندرہ": "pandrah",
    "سولہ": "solah",
    "سترہ": "satrah",
    "اٹھارہ": "atharah",
    "انیس": "unnees",
    "بیس": "bees",
}

# Aspirated consonant digraphs (Do-Chashmi Heh compounds)
_ASPIRATED_CONSONANTS: Final[dict[str, str]] = {
    "بھ": "bh",
    "پھ": "ph",
    "تھ": "th",
    "ٹھ": "th",
    "جھ": "jh",
    "چھ": "chh",
    "دھ": "dh",
    "ڈھ": "dh",
    "رھ": "rh",
    "ڑھ": "rh",
    "کھ": "kh",
    "گھ": "gh",
    "لہ": "lh",
    "مھ": "mh",
    "نھ": "nh",
}

# Individual character mapping for phonetic fallback
_URDU_LETTER_MAP: Final[dict[str, str]] = {
    "ا": "a",
    "آ": "aa",
    "ب": "b",
    "پ": "p",
    "ت": "t",
    "ٹ": "t",
    "ث": "s",
    "ج": "j",
    "چ": "ch",
    "ح": "h",
    "خ": "kh",
    "د": "d",
    "ڈ": "d",
    "ذ": "z",
    "ر": "r",
    "ڑ": "r",
    "ز": "z",
    "ژ": "zh",
    "س": "s",
    "ش": "sh",
    "ص": "s",
    "ض": "z",
    "ط": "t",
    "ظ": "z",
    "ع": "a",
    "غ": "gh",
    "ف": "f",
    "ق": "q",
    "ک": "k",
    "گ": "g",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ں": "n",
    "و": "o",
    "ہ": "h",
    "ھ": "h",
    "ء": "",
    "ئ": "y",
    "ی": "i",
    "ے": "e",
    "ؤ": "o",
    # Urdu digits
    "۰": "0",
    "۱": "1",
    "۲": "2",
    "۳": "3",
    "۴": "4",
    "۵": "5",
    "۶": "6",
    "۷": "7",
    "۸": "8",
    "۹": "9",
}

# Common grammatical suffixes to extract or match
_COMMON_SUFFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("سکتا", "sakta"),
    ("سکتی", "sakti"),
    ("سکتے", "sakte"),
    ("کرتا", "karta"),
    ("کرتی", "karti"),
    ("کرتے", "karte"),
    ("والا", "wala"),
    ("والی", "wali"),
    ("والے", "wale"),
    ("وں", "on"),
    ("یں", "ain"),
    ("ئے", "ye"),
    ("ئی", "i"),
    ("ائے", "aye"),
    ("ائی", "ai"),
    ("اؤ", "ao"),
    ("تے", "te"),
    ("تا", "ta"),
    ("تی", "ti"),
    ("نا", "na"),
    ("نے", "ne"),
    ("نی", "ni"),
)


def contains_urdu_script(text: str) -> bool:
    """Check if the string contains any Arabic or Urdu Unicode characters."""
    return bool(_URDU_RANGE_PATTERN.search(text))


def normalize_urdu_text(text: str) -> str:
    """Normalize Unicode variations and Arabic characters into standard Urdu."""
    normalized = unicodedata.normalize("NFKC", text)
    res: list[str] = []
    for ch in normalized:
        res.append(_CHAR_NORMALIZATION.get(ch, ch))
    return "".join(res)


def _transliterate_word_phonetic(word: str) -> str:
    """Phonetically transliterate any arbitrary Urdu word not in dictionary.
    
    Handles aspirated consonants, initial vs medial vs final vowels,
    and diphthongs to produce natural Roman Urdu spelling.
    """
    if not word:
        return ""

    if not contains_urdu_script(word):
        return word

    # Check common compound suffixes first (e.g. دیستکتا -> de sakta)
    for urdu_suffix, roman_suffix in _COMMON_SUFFIXES:
        if word.endswith(urdu_suffix) and len(word) > len(urdu_suffix):
            base = word[:-len(urdu_suffix)]
            base_roman = _URDU_TO_ROMAN_DICT.get(base)
            if not base_roman:
                base_roman = _transliterate_word_phonetic(base)
            return f"{base_roman} {roman_suffix}".strip()

    chars = list(word)
    n = len(chars)
    result: list[str] = []
    i = 0

    while i < n:
        # Check two-character aspirated consonants
        if i + 1 < n:
            two_chars = chars[i] + chars[i + 1]
            if two_chars in _ASPIRATED_CONSONANTS:
                result.append(_ASPIRATED_CONSONANTS[two_chars])
                i += 2
                continue
            # Special diphthong handling
            if two_chars == "ہو":
                result.append("ho")
                i += 2
                continue
            if two_chars == "ئی":
                result.append("i")
                i += 2
                continue
            if two_chars == "ئے":
                result.append("ye")
                i += 2
                continue
            if two_chars == "اؤ":
                result.append("ao")
                i += 2
                continue
            if two_chars == "وں":
                result.append("on")
                i += 2
                continue
            if two_chars == "یں":
                result.append("ein")
                i += 2
                continue

        ch = chars[i]

        # Initial vowel nuances
        if i == 0:
            if ch == "آ":
                result.append("aa")
                i += 1
                continue
            if ch == "ا":
                if i + 1 < n:
                    next_ch = chars[i + 1]
                    if next_ch == "و":
                        result.append("o")
                        i += 2
                        continue
                    if next_ch == "ی":
                        result.append("i")
                        i += 2
                        continue
                    if next_ch == "ے":
                        result.append("e")
                        i += 2
                        continue
                result.append("a")
                i += 1
                continue

        # Terminal vowel nuances
        if i == n - 1:
            if ch == "ے":
                result.append("e")
                i += 1
                continue
            if ch == "ی":
                result.append("i")
                i += 1
                continue
            if ch == "ہ":
                result.append("a")
                i += 1
                continue
            if ch == "ں":
                result.append("n")
                i += 1
                continue

        # Medial vowel nuances
        if ch == "ی":
            # If flanked by consonants, it usually acts as 'ee' or 'e'
            result.append("e" if (i == n - 2 and chars[-1] == "ں") else "i")
            i += 1
            continue
        if ch == "و":
            result.append("o")
            i += 1
            continue

        # Default letter mapping
        result.append(_URDU_LETTER_MAP.get(ch, ch))
        i += 1

    transliterated = "".join(result)
    # Clean up duplicate vowels or punctuation
    transliterated = re.sub(r"([aeiou])\1{2,}", r"\1\1", transliterated)
    return transliterated


def transliterate_urdu_to_roman(text: str) -> str:
    """Robust entry point to convert arbitrary Urdu text into clean Roman Urdu.
    
    1. Normalizes characters (standard Urdu Unicode).
    2. Uses word and multi-word dictionary matches for maximum precision on academic & conversational terms.
    3. Employs morphological and phonetic decomposition for unseen/out-of-vocabulary words.
    4. Preserves English acronyms, numbers, and existing Latin text.
    """
    if not text or not text.strip():
        return ""

    if not contains_urdu_script(text):
        return text.strip()

    normalized = normalize_urdu_text(text)

    # First pass: replace multi-word phrases from dictionary
    # Sort phrases by word length descending so longest matches match first
    multiword_keys = sorted(
        [k for k in _URDU_TO_ROMAN_DICT if " " in k],
        key=len,
        reverse=True,
    )
    for phrase in multiword_keys:
        if phrase in normalized:
            normalized = normalized.replace(phrase, _URDU_TO_ROMAN_DICT[phrase])

    # Second pass: tokenize remaining words and translate
    tokens = re.findall(r"[\w\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]+|[^\w\s]", normalized)
    output_tokens: list[str] = []

    for token in tokens:
        # If it's punctuation or pure numbers/Latin
        if not contains_urdu_script(token):
            output_tokens.append(token)
            continue

        clean = token.strip("،.؟!?؛:")
        if not clean:
            output_tokens.append(token)
            continue

        # Exact dictionary match
        if clean in _URDU_TO_ROMAN_DICT:
            roman_word = _URDU_TO_ROMAN_DICT[clean]
        else:
            # Phonetic fallback
            roman_word = _transliterate_word_phonetic(clean)

        # Restore punctuation if any
        if token.startswith("،") or token.startswith("؟") or token.startswith("."):
            roman_word = token[0] + roman_word
        if token.endswith("،") or token.endswith("؟") or token.endswith("."):
            roman_word = roman_word + token[-1]

        output_tokens.append(roman_word)

    # Join tokens into a sentence
    sentence = " ".join(output_tokens)

    # Post-processing fixes for Whisper acoustic mishearings
    typo_fixes = {
        r"\bsbbt\b": "submit",
        r"\bsbmt\b": "submit",
        r"\bagzan\b": "exam",
        r"\bagzam\b": "exam",
        r"\baynosty\b": "university",
        r"\bfys\b": "fees",
        r"\bdystkta\b": "de sakta",
        r"\bkroa\b": "karwa",
    }
    for pattern, repl in typo_fixes.items():
        sentence = re.sub(pattern, repl, sentence, flags=re.IGNORECASE)

    # Fix spacing around punctuation
    sentence = re.sub(r"\s+([,\.?!;:])", r"\1", sentence)
    # Remove any stray Urdu characters that might have escaped
    sentence = re.sub(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]", "", sentence)
    # Collapse multiple spaces
    sentence = re.sub(r"\s{2,}", " ", sentence).strip()

    return sentence


_PHONETIC_NORMALIZATION: Final[dict[str, str]] = {
    r"\bsabitak\b": "abhi tak",
    r"\bsabhy tk\b": "abhi tak",
    r"\babhy tk\b": "abhi tak",
    r"\bsabit\b": "submit",
    r"\bsbbt\b": "submit",
    r"\bsbmt\b": "submit",
    r"\bniye\b": "nahi",
    r"\bnhyn\b": "nahi",
    r"\bseta\b": "sakta",
    r"\bde seta\b": "de sakta",
    r"\bdystkta\b": "de sakta",
    r"\bagzan\b": "exam",
    r"\bagzam\b": "exam",
    r"\baynosty\b": "university",
    r"\bfys\b": "fees",
    r"\bfee\b": "fees",
    r"\bkroa\b": "karwa",
    r"\bkrwa\b": "karwa",
    r"\bkye\b": "kiye",
    r"\bmyn\b": "main",
    r"\baoy\b": "abhi",
    r"hoئy": "hui",
    r"\bkiya\b": "kya",
}


def clean_and_normalize_roman_urdu(text: str) -> str:
    """Take any transcribed text (Urdu script or imperfect Roman Urdu) and make it clean Roman Urdu."""
    if not text:
        return ""

    # 1. Clean repeating phrase hallucinations (e.g. "sa da sa da sa da")
    text = re.sub(r"(\b(?:\w+\s+){1,3}\w+)\b(?:\s+\1\b){2,}", r"\1", text, flags=re.IGNORECASE)
    # Strip repeating words at tail
    words = text.split()
    while len(words) >= 4 and words[-2:] == words[-4:-2]:
        words = words[:-2]
    # Remove nonsensical trailing repeats like "sa da"
    if len(words) >= 3 and words[-2:] in (["sa", "da"], ["da", "sa"], ["aye", "aye"], ["ayk", "ayk"]):
        words = words[:-2]
    text = " ".join(words)

    # 2. If Urdu script exists, transliterate it first
    if contains_urdu_script(text):
        text = transliterate_urdu_to_roman(text)

    # 3. Clean common phonetic Whisper speech-to-text artifacts
    for pattern, repl in _PHONETIC_NORMALIZATION.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # 4. Collapse spaces and clean punctuation
    text = re.sub(r"\s+([,\.?!;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

