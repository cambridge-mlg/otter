import random
from datetime import datetime


def _generate_experiment_name() -> str:
    adjectives = [
        "swift",
        "fierce",
        "gentle",
        "clever",
        "brave",
        "cunning",
        "loyal",
        "playful",
        "quiet",
        "bold",
        "majestic",
        "nimble",
        "mighty",
        "elegant",
        "sly",
        "energetic",
        "daring",
        "graceful",
        "wild",
        "fearless",
    ]

    animals = [
        "aardvark",
        "tiger",
        "eagle",
        "wolf",
        "panther",
        "falcon",
        "fox",
        "bear",
        "lion",
        "cheetah",
        "hawk",
        "otter",
        "deer",
        "jaguar",
        "leopard",
        "owl",
        "raven",
        "buffalo",
        "cougar",
        "lynx",
        "badger",
    ]

    adjective = random.choice(adjectives)
    animal = random.choice(animals)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    return f"{timestamp}-{adjective}-{animal}"
