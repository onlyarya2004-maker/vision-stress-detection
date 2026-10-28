"""
Sentences the CLIP "hint" model compares each camera frame against
(context_hints.py). Pure data, shared by the one-time conversion script
(scripts/convert_clip_to_onnx.py, which turns them into text embeddings) and
by the app, so the two can never drift apart. Editing any sentence means
re-running that script.

Hints are SUPPORT for the scene model, never a decision on their own. Each
group's probability is the share of CLIP's softmax over ALL sentences below
that falls on that group, so the neutral "other activity" sentences compete
with the context ones.
"""

HINT_PROMPTS = {
    'Studying': [
        "a photo of a student writing in a notebook at a desk",
        "a photo of a person studying with books and papers",
        "a photo of a school student in a school uniform at a desk",
        "a photo of a college student studying with a laptop and notebooks",
        "a photo of students sitting on benches in a classroom",
        "a photo of a person reading a book",
    ],
    'Hospital': [
        "a photo of a doctor wearing a white coat or scrubs",
        "a photo of a patient lying in a hospital bed",
        "a photo of a nurse in a hospital",
        "a photo of a person in a hospital room with medical machines",
        "a photo of a person wearing a surgical mask in a clinic",
    ],
    'Vehicle': [
        "a photo of a person sitting inside a car",
        "a photo of a person driving a car",
        "a photo of a person inside a truck cab",
        "a photo of the view from the driver seat of a vehicle",
        "a photo of a passenger on a bus",
    ],
    'Other': [
        "a photo of a person relaxing on a sofa",
        "a photo of a person using a phone",
        "a photo of a person in a bedroom",
        "a photo of a person playing video games",
        "a photo of a person in a cafe",
        "a photo of people in an office meeting",
        "a photo of a person eating a meal",
        "a photo of a person watching television",
        "a photo of a person in a living room",
        "a photo of a person talking on a video call",
    ],
}
