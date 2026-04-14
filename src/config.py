"""
Central configuration for the GACA sentiment pipeline.
Update BATCH_DATE before each new run.
"""

from datetime import date

# ── Run config ────────────────────────────────────────────────────────────────
BATCH_DATE  = '2026-04-13'
MODEL_NAME  = 'gemini-2.5-flash'

# ── Paths (relative to project root) ─────────────────────────────────────────
TAXONOMY_PATH = 'src/taxonomy_generation/topic_subtopic_refined.csv'

# Raw input directories — drop scraper/source files here, no path changes needed
RAW_TRIPADVISOR_DIR = 'raw_data/tripadvisor'
RAW_GOOGLE_DIR      = 'raw_data/google_maps'
RAW_X_DIR           = 'raw_data/x_data'

# Processed outputs — written by data_prep scripts, read by sentiment pipeline
TRIPADVISOR_PATH = 'tripadvisor_data/tripadvisor_jan2025_mar2026.csv'
GOOGLE_PATH      = 'Google_review_data/airport_google_reviews_jan2025_mar2026.csv'
X_COMBINED_PATH  = 'X_data/final_combined_x_data.csv'
X_BATCH_DIR     = f'X_data/intermediate_X_sentiments/{BATCH_DATE}_batches'

# Airline intermediate + output
AIRLINE_BATCH_DIR = f'airlines_sentiment/intermediate_sentiments/{BATCH_DATE}_batches'
AIRLINE_OUTPUT    = f'airlines_sentiment/final_airline_sentiment_{date.today()}.csv'
AIRLINE_X_OUTPUT  = f'airlines_sentiment/final_x_airline_sentiment_{date.today()}.csv'

# Airport intermediate + output
AIRPORT_BATCH_DIR = f'airport_sentiment/intermediate_sentiments/{BATCH_DATE}_batches'
AIRPORT_OUTPUT    = f'airport_sentiment/final_airport_sentiment_{date.today()}.csv'
AIRPORT_X_OUTPUT  = f'airport_sentiment/final_x_airport_sentiment_{date.today()}.csv'

# ── Language map ──────────────────────────────────────────────────────────────
LANG_MAP = {
    'en': 'English',   'ar': 'Arabic',     'es': 'Spanish',    'fr': 'French',
    'de': 'German',    'it': 'Italian',    'pt': 'Portuguese', 'tr': 'Turkish',
    'ru': 'Russian',   'zh': 'Chinese',    'ja': 'Japanese',   'ko': 'Korean',
    'nl': 'Dutch',     'ur': 'Urdu',       'fa': 'Persian',    'hi': 'Hindi',
    'ro': 'Romanian',  'hu': 'Hungarian',  'pl': 'Polish',     'sv': 'Swedish',
    'da': 'Danish',    'fi': 'Finnish',    'no': 'Norwegian',  'id': 'Indonesian',
    'ms': 'Malay',     'th': 'Thai',       'af': 'Afrikaans',  'ca': 'Catalan',
    'cy': 'Welsh',     'el': 'Greek',      'bn': 'Bengali',    'fil': 'Filipino',
    'hr': 'Croatian',  'iw': 'Hebrew',     'so': 'Somali',     'uz': 'Uzbek',
    'zh-hant': 'Chinese (Traditional)',
}

# ── X entity keyword rules ────────────────────────────────────────────────────
AIRLINE_ENTITY_KEYWORDS = {
    'saudia'   : ['saudia', 'saudi airlines', 'saudi arabian airlines', 'الخطوط السعودية',
                  'سعودية', 'خطوط سعودية', 'الخطوط', '@saudiairlines', '@saudi_airlines',
                  'saudi_airlines', 'sv', 'saudiairlines'],
    'flynas'   : ['flynas', 'fly nas', 'طيران ناس', 'فلاي ناس', 'ناس', '@flynas', 'nas air'],
    'flyadeal' : ['flyadeal', 'fly adeal', 'طيران أديل', 'طيران اديل', 'فلاي ديل',
                  'أديل', 'اديل', '@flyadeal'],
}
AIRPORT_ENTITY_KEYWORDS = {
    'RUH': ['riyadh', 'الرياض', 'مطار الرياض', 'الملك خالد', 'king khalid', 'ruh',
            'مطار خالد', 'Terminal 4', 'تيرمينال', '@kkia', '@riyadhairport'],
    'JED': ['jeddah', 'jedda', 'جدة', 'مطار جدة', 'الملك عبدالعزيز',
            'king abdulaziz', 'jed', 'مطار عبدالعزيز', '@kaia_jeddah'],
    'DMM': ['dammam', 'الدمام', 'مطار الدمام', 'الملك فهد', 'king fahd',
            'dmm', 'مطار فهد', '@kfiairport', 'kfia'],
}
AIRPORT_GENERIC_KEYWORDS = [
    'مطار', 'airport', 'terminal', 'صالة', 'صالة المغادرة', 'صالة الوصول',
    'وصول', 'مغادرة', 'arrivals', 'departures', 'gate', 'بوابة',
    'موقف', 'parking', 'جوازات', 'immigration', 'جمارك', 'customs',
    'check-in', 'تسجيل وصول', 'baggage claim', 'استلام الامتعة',
]
AIRLINE_GENERIC_KEYWORDS = [
    'flight', 'رحلة', 'طيران', 'airline', 'خطوط', 'تذكرة', 'ticket',
    'حجز', 'booking', 'boarding', 'صعود', 'delay', 'تأخير',
    'luggage', 'baggage', 'امتعة', 'شنطة', 'crew', 'طاقم',
    'captain', 'كابتن', 'seat', 'مقعد', 'class', 'درجة',
    'compensation', 'تعويض', 'refund', 'استرداد',
]

# ── Date filter ───────────────────────────────────────────────────────────────
MIN_DATE = '2025-11-01'
