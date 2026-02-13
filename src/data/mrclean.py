import re

# Special tokens
START_TOKEN = '<s>'
END_TOKEN = '</s>'
PADDING_TOKEN = '<pad>'

# Builds a pad sequence like: </s><pad><pad>...<pad>
def _make_padding_sequence(seq_length):
    return ''.join([END_TOKEN] + seq_length * [PADDING_TOKEN])

# Cleans up extra spacing and spacing before punctuation
def cleanup_extra_spaces(text):
    multiple_spaces_ex = re.compile(r'[ \t\u00A0]+')
    space_before_punctuation_ex = re.compile(r'[ \t\u00A0]([.,;!?])')
    text = multiple_spaces_ex.sub(' ', text)
    text = space_before_punctuation_ex.sub(r'\1', text)
    return text.strip()

# Generic cleanup applicable to most datasets
def generic_cleanup(text, seq_length, add_start_end=True, insert_pad_after_doc=True):
    text = cleanup_extra_spaces(text)

    if add_start_end:
        text = START_TOKEN + text

    if insert_pad_after_doc:
        pad_seq = _make_padding_sequence(seq_length)
        # Treat double newlines as document separators
        text = re.sub(r'\n\n+', pad_seq + START_TOKEN, text)
        text += _make_padding_sequence(seq_length)

    return text

# Dataset-specific wrappers (if needed)
def cleanup_simple_wiki(text, seq_length):
    return generic_cleanup(text, seq_length)

def cleanup_tiny_stories(text, seq_length):
    return generic_cleanup(text, seq_length)

def cleanup_bnc_spoken(text, seq_length):
    return generic_cleanup(text, seq_length)

def cleanup_childes(text, seq_length):
    return generic_cleanup(text, seq_length)

def cleanup_gutenberg(text, seq_length):
    return generic_cleanup(text, seq_length, add_start_end=False)

def cleanup_open_subtitles(text, seq_length):
    subtitle_credit_ex = re.compile(r'^.*subtitle.*$\n', re.MULTILINE | re.IGNORECASE)
    text = subtitle_credit_ex.sub('', text)
    return generic_cleanup(text, seq_length)

def cleanup_switchboard(text, seq_length):
    return generic_cleanup(text, seq_length, add_start_end=False)
