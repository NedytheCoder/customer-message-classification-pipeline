from pathlib import Path


class Config:
    # Input text columns
    TICKET_SUMMARY = "Ticket Summary"
    TICKET_ID = "Ticket id"
    INTERACTION_CONTENT = "Interaction content"

    TEXT_COLUMNS = [TICKET_SUMMARY, INTERACTION_CONTENT]

    # Label columns after renaming original Type 1-Type 4 columns
    TYPE_COLS = ["y1", "y2", "y3", "y4"]

    # Core assessment target label.
    CLASS_COL = "y2"

    # Used by the existing prototype to run separate experiments per Type 1 group.
    GROUPED = "y1"

    # Minority target labels
    SUBLABELS = ["y3", "y4"]

    # Reproducibility: single source of truth for the random seed.
    SEED = 0

    # Paths (centralised here instead of hardcoded in each module).
    DATA_DIR = Path("data")
    INPUT_FILES = ["AppGallery.csv", "Purchasing.csv"]
    ARTIFACT_DIR = Path("artifacts")
    OUTPUT_DIR = Path("outputs")
    NEW_MESSAGES = DATA_DIR / "new_messages.csv"
