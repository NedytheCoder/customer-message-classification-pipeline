import re
import pandas as pd
from config import Config
import html

company_templates = [
    # english
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team\,?",
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is a company incorporated under the laws of Ireland with its headquarters in Dublin, Ireland\.?",
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is the provider of Huawei Mobile Services to Huawei and Honor device owners in (?:Europe|\*\*\*\*\*\(LOC\)), Canada, Australia, New Zealand and other countries\.?",
    # german
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Kundenservice\,?",
    r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist eine Gesellschaft nach irischem Recht mit Sitz in Dublin, Irland\.?",
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist der Anbieter von Huawei Mobile Services für Huawei- und Honor-Gerätebesitzer in Europa, Kanada, Australien, Neuseeland und anderen Ländern\.?",
    # french
    r"L'équipe d'assistance à la clientèle d'Aspiegel\,?",
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est une société de droit irlandais dont le siège est à Dublin, en Irlande\.?",
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est le fournisseur de services mobiles Huawei aux propriétaires d'appareils Huawei et Honor en Europe, au Canada, en Australie, en Nouvelle-Zélande et dans d'autres pays\.?",
    # spanish
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Soporte Servicio al Cliente\,?",
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) es una sociedad constituida en virtud de la legislación de Irlanda con su sede en Dublín, Irlanda\.?",
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE es el proveedor de servicios móviles de Huawei a los propietarios de dispositivos de Huawei y Honor en Europa, Canadá, Australia, Nueva Zelanda y otros países\.?",
    # italian
    r"Il tuo team ad (?:Aspiegel|\*\*\*\*\*\(PERSON\)),?",
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è una società costituita secondo le leggi irlandesi con sede a Dublino, Irlanda\.?",
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è il fornitore di servizi mobili Huawei per i proprietari di dispositivi Huawei e Honor in Europa, Canada, Australia, Nuova Zelanda e altri paesi\.?",
    # portuguese
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team,?",
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é uma empresa constituída segundo as leis da Irlanda, com sede em Dublin, Irlanda\.?",
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é o provedor de Huawei Mobile Services para Huawei e Honor proprietários de dispositivos na Europa, Canadá, Austrália, Nova Zelândia e outros países\.?",
]

noise_patterns = [
    r"(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(re?\s*:)",
    r"[\[\]]",
    r"aspiegel support issue submit",
    r"\bnull\b|\bnan\b",
    r"support\.pt 自动回复:",
    r"(from\s*:)|(subject\s*:)|(sent\s*:)",
    r"(january|february|march|april|may|june|july|august|september|october|november|december)",
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
    r"\d{2}[:.]\d{2}",
    r"(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))",
    r"dear ((customer)|(user))",
    r"dear",
    r"(hello)|(hallo)|(hi )|(hi there)",
    r"good morning",
    r"thank you for your patience ((during (our)? investigation)|(and cooperation))?",
    r"thank you for contacting us",
    r"thank you for your availability",
    r"thank you for providing us this information",
    r"thank you for contacting",
    r"thank you for reaching us (back)?",
    r"thank you for patience",
    r"thank you for (your)? reply",
    r"thank you for (your)? response",
    r"thank you for (your)? cooperation",
    r"thank you for providing us with more information",
    r"thank you very kindly",
    r"thank you( very much)?",
    r"i would like to follow up on the case you raised on( the date)?",
    r"i will do my very best to assist you",
    r"in order to give you the best solution",
    r"could you please clarify your request with following information:",
    r"in this matter",
    r"we hope you(( are)|('re)) doing ((fine)|(well))",
    r"we apologize for the inconvenience",
    r"sent from my huawei (cell )?phone",
    r"original message",
    r"customer support team",
    r"\d+",            # any remaining digits
    r"[^\w\s]+",       # strip punctuation/symbols only; keep unicode letters (any language)
    r"(\s|^).(\s|$)",  # strip stray single characters
]

company_signature_detector = re.compile("|".join(company_templates), flags=re.IGNORECASE)
noise_pattern_detector = [re.compile(pattern, flags=re.IGNORECASE) for pattern in noise_patterns]
white_space_detector = re.compile(r"\s+")

def normalize_text(text):
    text = text.lower()
    text = company_signature_detector.sub(" ", text)
    for pattern in noise_pattern_detector:
        text = pattern.sub(" ", text)
    text = white_space_detector.sub(" ", text)
    return text.strip()

def normalize_text_columns(df, columns=Config.TEXT_COLUMNS):
    df = df.copy()
    for column in columns:
        df[column] = df[column].apply(normalize_text)
    return df

email_split_patterns = [
    r"From\s?:\s?xxxxx@xxxx\.com Sent\s?:.{30,70}Subject\s?:",
    r"On.{30,60}wrote:",
    r"(?:Re\s?:|RE\s?:)",
    r"\*\*\*\*\*\(PERSON\) Support issue submit",
    r"(?:\s?\*\*\*\*\*\(PHONE\))+$",
]
email_splitter = re.compile("|".join(email_split_patterns), flags=re.IGNORECASE)

def dedup(df, interaction_content_col=Config.INTERACTION_CONTENT, ticket_col=Config.TICKET_ID):
    if ticket_col not in df.columns:
        raise KeyError(
            f"This function needs a '{ticket_col}' column"
        )
     
    df = df.copy()
    deduped = pd.Series(index=df.index, dtype="object")

    for ticket_id, group in df.groupby(ticket_col):
            seen = set()
            for idx, content in group[interaction_content_col].items():
                kept = []
                for part in email_splitter.split(content):
                    if not part:
                        continue
                    part = email_splitter.sub("", part).strip()
                    if part and part not in seen:
                        seen.add(part)
                        kept.append(part)
                deduped.at[idx] = " ".join(kept)

    df[interaction_content_col] = deduped
    return df

def clean_labels(df):
    df = df.copy()
    merging_similar_labels_map = {"Others": "Other"}
    for col in Config.TYPE_COLS:
        if col not in df.columns:
            continue
        cleaned = (df[col].astype("string").map(lambda v: html.unescape(v) if isinstance(v, str) else v).str.strip().replace(merging_similar_labels_map))
        if col in Config.SUBLABELS:
            cleaned = cleaned.fillna("Other").replace({"": "Other"})
        df[col] = cleaned
    return df

def filter_rows_with_missing_labels(df, y2_label=Config.CLASS_COL):
    old_len = len(df)
    df = df.copy()

    df = df[df[y2_label].notna() & (df[Config.CLASS_COL].astype(str).str.strip() != "")]

    print(
        f"Filtered out {old_len - len(df)} rows with missing or empty "
        f"'{Config.CLASS_COL}' labels."
    )

    return df

def preprocess_training(df):
    df = dedup(df)
    df = normalize_text_columns(df)
    df = clean_labels(df)
    df = filter_rows_with_missing_labels(df)
    return df


def preprocess_inference(df):
    return normalize_text_columns(df)