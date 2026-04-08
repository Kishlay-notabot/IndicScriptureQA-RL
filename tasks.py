"""
Task definitions for IndicScriptureQA.

Each scenario now carries *structural metadata* alongside factual ground truth.
The grader evaluates BOTH factual accuracy AND semantic structure:
  - required_terms     → Sanskrit/domain terms the answer must use
  - required_sections  → conceptual aspects that must be covered
  - expected_order     → logical ordering of concepts
  - banned_terms       → misconception markers (penalty if present)

Difficulty controls:
  easy   → blatant factual error OR correct answer; max 5 steps
  medium → partial errors + missing citations + poor structure; max 8 steps
  hard   → subtle hallucinations + jumbled structure + terminology misuse; max 12 steps
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

from models import StructuralMeta


@dataclasses.dataclass
class Scenario:
    question: str
    given_answer: str
    ground_truth_answer: str
    ground_truth_citations: List[str]
    available_passages: List[str]
    answer_is_correct: bool            # overall (facts + structure)
    factual_is_correct: bool           # facts alone
    structural_meta: StructuralMeta
    structural_hints: List[str]        # non-spoiler hints shown to agent


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — verify-factual  (Easy)
# ═══════════════════════════════════════════════════════════════════════════════

VERIFY_FACTUAL_SCENARIOS: List[Scenario] = [
    Scenario(
        question="Who killed Ravana in the Ramayana?",
        given_answer="Lakshmana killed Ravana with a divine arrow during the battle of Lanka.",
        ground_truth_answer="Rama killed Ravana using the Brahmastra during the battle of Lanka.",
        ground_truth_citations=["Valmiki Ramayana, Yuddha Kanda, Sarga 108"],
        available_passages=[
            "Yuddha Kanda 108: Rama, following Agastya's counsel, invoked the Brahmastra and struck Ravana in the chest, ending his life.",
            "Yuddha Kanda 101: Lakshmana fought Indrajit and defeated him but did not fight Ravana directly in the final duel.",
        ],
        answer_is_correct=False,
        factual_is_correct=False,
        structural_meta=StructuralMeta(
            required_terms=["Rama", "Ravana", "Brahmastra"],
            required_sections=["who performed the act", "weapon used", "context of battle"],
            expected_order=["context of battle", "weapon used", "outcome"],
            banned_terms=[],
        ),
        structural_hints=["Answer should identify the warrior, the weapon, and the battle context."],
    ),
    Scenario(
        question="How many chapters does the Bhagavad Gita contain?",
        given_answer="The Bhagavad Gita contains 18 chapters, each called an Adhyaya.",
        ground_truth_answer="The Bhagavad Gita contains 18 chapters (Adhyayas), comprising 700 verses.",
        ground_truth_citations=["Bhagavad Gita, Chapters 1–18"],
        available_passages=[
            "The Bhagavad Gita is a 700-verse scripture that is part of the Mahabharata (Bhishma Parva, chapters 25–42). It consists of 18 chapters.",
        ],
        answer_is_correct=True,
        factual_is_correct=True,
        structural_meta=StructuralMeta(
            required_terms=["Adhyaya", "18"],
            required_sections=["chapter count", "structure note"],
            expected_order=[],
            banned_terms=[],
        ),
        structural_hints=["Mention chapter count and the term for chapters."],
    ),
    Scenario(
        question="Who narrated the Bhagavad Gita to Arjuna?",
        given_answer="Vyasa narrated the Bhagavad Gita to Arjuna on the battlefield of Kurukshetra.",
        ground_truth_answer="Krishna narrated the Bhagavad Gita to Arjuna on the battlefield of Kurukshetra.",
        ground_truth_citations=["Bhagavad Gita 1.1", "Bhagavad Gita 2.11"],
        available_passages=[
            "Bhagavad Gita 2.11: Sri Bhagavan (Krishna) said — You grieve for those who should not be grieved for.",
            "Vyasa composed the Mahabharata and dictated it to Ganesha but was not the speaker of the Gita.",
        ],
        answer_is_correct=False,
        factual_is_correct=False,
        structural_meta=StructuralMeta(
            required_terms=["Krishna", "Arjuna", "Kurukshetra"],
            required_sections=["speaker", "listener", "setting"],
            expected_order=["speaker", "listener", "setting"],
            banned_terms=[],
        ),
        structural_hints=["Identify the speaker, the listener, and the setting."],
    ),
    Scenario(
        question="What is the first word of the Rigveda?",
        given_answer="The first word of the Rigveda is 'Agnim' (अग्निम्), invoking the fire deity Agni.",
        ground_truth_answer="The first word of the Rigveda is 'Agnim' (अग्निम्), the accusative form of Agni, beginning the hymn to the fire deity.",
        ground_truth_citations=["Rigveda 1.1.1"],
        available_passages=[
            "Rigveda 1.1.1: Agnim ile purohitam yajnasya devam ritvijam — I praise Agni, the foremost priest, the divine minister of the sacrifice.",
        ],
        answer_is_correct=True,
        factual_is_correct=True,
        structural_meta=StructuralMeta(
            required_terms=["Agnim", "Agni", "Rigveda"],
            required_sections=["the word", "its meaning", "its significance"],
            expected_order=["the word", "its meaning"],
            banned_terms=[],
        ),
        structural_hints=["State the word, explain its grammatical form, and note its significance."],
    ),
    Scenario(
        question="In the Mahabharata, who was the commander-in-chief of the Kaurava army on the first day?",
        given_answer="Drona was the first commander-in-chief of the Kaurava army at Kurukshetra.",
        ground_truth_answer="Bhishma was the first commander-in-chief of the Kaurava army, leading from days 1 through 10.",
        ground_truth_citations=["Mahabharata, Bhishma Parva"],
        available_passages=[
            "Bhishma Parva: Bhishma was appointed supreme commander of the Kaurava forces. He led the army for the first ten days of the war.",
            "Drona Parva: After Bhishma fell, Drona was appointed the second commander-in-chief.",
        ],
        answer_is_correct=False,
        factual_is_correct=False,
        structural_meta=StructuralMeta(
            required_terms=["Bhishma", "Kaurava"],
            required_sections=["commander identity", "duration of command"],
            expected_order=["commander identity", "duration of command"],
            banned_terms=[],
        ),
        structural_hints=["Identify the commander and state how long they held command."],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — correct-and-cite  (Medium)
#   Scenarios now include structural problems: missing aspects, wrong ordering,
#   imprecise terminology, alongside the citation/factual issues.
# ═══════════════════════════════════════════════════════════════════════════════

CORRECT_AND_CITE_SCENARIOS: List[Scenario] = [
    Scenario(
        question="What does Krishna say about Karma Yoga in the Bhagavad Gita?",
        given_answer="Krishna tells Arjuna to perform his duty without attachment to results. He says action is superior to inaction and that one should work selflessly.",
        ground_truth_answer="Krishna teaches that one has the right to perform prescribed duties but is not entitled to the fruits of actions. He advocates nishkama karma — selfless action without attachment to outcomes — as the path to liberation.",
        ground_truth_citations=["Bhagavad Gita 2.47", "Bhagavad Gita 3.19"],
        available_passages=[
            "Bhagavad Gita 2.47: Karmanye vadhikaraste ma phaleshu kadachana — You have the right to perform your duty, but you are not entitled to the fruits of your actions.",
            "Bhagavad Gita 3.19: Therefore, without attachment, always perform the work that has to be done; for by performing work without attachment, one attains the Supreme.",
            "Bhagavad Gita 3.4: Not by merely abstaining from action can one achieve freedom from reaction, nor by renunciation alone can one attain perfection.",
        ],
        answer_is_correct=False,
        factual_is_correct=False,
        structural_meta=StructuralMeta(
            required_terms=["nishkama karma", "Karma Yoga", "phala"],
            required_sections=["core teaching", "key verse reference", "philosophical implication"],
            expected_order=["core teaching", "key verse reference", "philosophical implication"],
            banned_terms=[],
        ),
        structural_hints=[
            "Use the Sanskrit term for selfless action.",
            "Structure: teaching → supporting verse → implication for liberation.",
        ],
    ),
    Scenario(
        question="What are the four Mahavakyas of the Upanishads?",
        given_answer="The four Mahavakyas are: Prajnanam Brahma, Aham Brahmasmi, Tat Tvam Asi, and Ayam Atma Brahma. They express the identity of the self with Brahman.",
        ground_truth_answer="The four Mahavakyas are: 'Prajnanam Brahma' (Consciousness is Brahman) from Aitareya Upanishad, 'Aham Brahmasmi' (I am Brahman) from Brihadaranyaka Upanishad, 'Tat Tvam Asi' (Thou art That) from Chandogya Upanishad, and 'Ayam Atma Brahma' (This Self is Brahman) from Mandukya Upanishad.",
        ground_truth_citations=[
            "Aitareya Upanishad 3.3",
            "Brihadaranyaka Upanishad 1.4.10",
            "Chandogya Upanishad 6.8.7",
            "Mandukya Upanishad 1.2",
        ],
        available_passages=[
            "Aitareya Upanishad 3.3: Prajnanam Brahma — Consciousness is Brahman. This Mahavakya belongs to the Rigveda.",
            "Brihadaranyaka Upanishad 1.4.10: Aham Brahmasmi — I am Brahman. This declaration belongs to the Yajurveda.",
            "Chandogya Upanishad 6.8.7: Tat Tvam Asi — Thou art That. Uddalaka teaches this to Shvetaketu. Belongs to the Samaveda.",
            "Mandukya Upanishad 1.2: Ayam Atma Brahma — This Self is Brahman. Belongs to the Atharvaveda.",
        ],
        answer_is_correct=False,
        factual_is_correct=True,  # the four names are correct; missing source attribution
        structural_meta=StructuralMeta(
            required_terms=["Mahavakya", "Brahman", "Atman"],
            required_sections=["each vakya with translation", "source Upanishad for each", "unifying theme"],
            expected_order=["vakya list", "unifying theme"],
            banned_terms=[],
        ),
        structural_hints=[
            "Each Mahavakya should be paired with its source Upanishad.",
            "Conclude with the unifying Advaita theme.",
        ],
    ),
    Scenario(
        question="Describe the concept of Dharma as explained in the Mahabharata.",
        given_answer="Dharma in the Mahabharata is presented as a complex moral code. Bhishma explains dharma extensively while lying on the bed of arrows. The text suggests dharma is subtle and context-dependent.",
        ground_truth_answer="The Mahabharata presents dharma as subtle (sukshma) and context-dependent. Bhishma's discourse on dharma in the Shanti Parva and Anushasana Parva covers duties of kings, ethics of war, and personal righteousness. The famous dictum states: 'Dharma is that which sustains all beings.'",
        ground_truth_citations=["Mahabharata, Shanti Parva", "Mahabharata, Anushasana Parva"],
        available_passages=[
            "Shanti Parva: Bhishma, lying on the bed of arrows, instructs Yudhishthira on rajadharma (duties of kings), moksha-dharma, and apaddharma (ethics in emergencies).",
            "Anushasana Parva: Continuation of Bhishma's teachings on dana (charity), dharma, and moral conduct.",
            "Vana Parva 313.117: Dharmo rakshati rakshitah — Dharma protects those who protect dharma.",
        ],
        answer_is_correct=False,
        factual_is_correct=True,
        structural_meta=StructuralMeta(
            required_terms=["sukshma", "rajadharma", "apaddharma", "dharma"],
            required_sections=["definition of dharma", "Bhishma's discourse topics", "key dictum"],
            expected_order=["definition of dharma", "Bhishma's discourse topics", "key dictum"],
            banned_terms=[],
        ),
        structural_hints=[
            "Use the Sanskrit term for dharma's subtlety.",
            "Cover the three categories Bhishma teaches: rajadharma, moksha-dharma, apaddharma.",
        ],
    ),
    Scenario(
        question="What is the significance of the number 108 in Hindu scriptures?",
        given_answer="The number 108 is sacred because there are 108 Upanishads in the Muktika canon. Japa malas have 108 beads for chanting.",
        ground_truth_answer="108 is considered sacred in Hinduism for multiple reasons: the Muktika canon lists 108 Upanishads, japa malas contain 108 beads, there are 108 names (ashtottara) for major deities, and mathematically 1×(2²)×(3³)=108. The distance between the Earth and Sun is approximately 108 times the Sun's diameter.",
        ground_truth_citations=["Muktika Upanishad 1.30–39"],
        available_passages=[
            "Muktika Upanishad: Rama narrates 108 Upanishads to Hanuman. The list includes major (mukhya) and minor Upanishads.",
            "In Hindu tradition, 108 appears in mala beads, temple steps, and as the count of names in ashtottara-namavali (108 names of deities like Vishnu, Shiva, Lakshmi).",
        ],
        answer_is_correct=False,
        factual_is_correct=True,
        structural_meta=StructuralMeta(
            required_terms=["ashtottara", "Muktika", "japa"],
            required_sections=["scriptural significance", "ritual significance", "mathematical note"],
            expected_order=["scriptural significance", "ritual significance"],
            banned_terms=[],
        ),
        structural_hints=[
            "Cover scriptural, ritual, AND mathematical dimensions.",
            "Use the term 'ashtottara-namavali' for the 108-name tradition.",
        ],
    ),
    Scenario(
        question="Who composed the Yoga Sutras?",
        given_answer="Patanjali composed the Yoga Sutras, a foundational text of Raja Yoga, consisting of 196 sutras organized into four padas.",
        ground_truth_answer="Patanjali composed the Yoga Sutras, a foundational text of Raja Yoga comprising 196 sutras in four padas: Samadhi Pada, Sadhana Pada, Vibhuti Pada, and Kaivalya Pada.",
        ground_truth_citations=["Yoga Sutras of Patanjali 1.1"],
        available_passages=[
            "Yoga Sutras 1.1: Atha yoga-anushasanam — Now, the exposition of Yoga is being made.",
            "The four padas are: Samadhi Pada (51 sutras), Sadhana Pada (55 sutras), Vibhuti Pada (56 sutras), and Kaivalya Pada (34 sutras).",
        ],
        answer_is_correct=True,
        factual_is_correct=True,
        structural_meta=StructuralMeta(
            required_terms=["Patanjali", "Raja Yoga", "pada"],
            required_sections=["author", "text structure", "pada names"],
            expected_order=["author", "text structure", "pada names"],
            banned_terms=[],
        ),
        structural_hints=["Name all four padas explicitly."],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3 — fix-hallucination  (Hard)
#   Includes subtle factual errors PLUS structural/terminological problems:
#   wrong ordering of concepts, misused Sanskrit terms, incomplete coverage.
# ═══════════════════════════════════════════════════════════════════════════════

FIX_HALLUCINATION_SCENARIOS: List[Scenario] = [
    Scenario(
        question="Explain the Dashavatara (ten avatars) of Vishnu as described in the Puranas.",
        given_answer="The ten avatars of Vishnu are: Matsya (fish), Kurma (tortoise), Varaha (boar), Narasimha (man-lion), Vamana (dwarf), Parashurama, Rama, Balarama, Buddha, and Kalki. Each avatar appeared in a specific yuga to restore cosmic order. Narasimha appeared in Treta Yuga to defeat the demon Hiranyakashipu. Vamana tricked the demon king Ravana by asking for three steps of land.",
        ground_truth_answer="The ten avatars of Vishnu are: Matsya, Kurma, Varaha, Narasimha, Vamana, Parashurama, Rama, Krishna (or Balarama in some lists), Buddha, and Kalki. Narasimha appeared in Satya Yuga (not Treta) to defeat Hiranyakashipu. Vamana tricked the demon king Mahabali (not Ravana) by asking for three paces of land. The avatars follow an evolutionary sequence reflecting increasing biological complexity.",
        ground_truth_citations=[
            "Bhagavata Purana 1.3",
            "Garuda Purana 1.86",
            "Bhagavata Purana 7.8 (Narasimha)",
            "Bhagavata Purana 8.18–22 (Vamana)",
        ],
        available_passages=[
            "Bhagavata Purana 1.3: Lists the avatars of Vishnu including Matsya, Kurma, Varaha, Narasimha, Vamana, Parashurama, Rama, Krishna, Buddha, and Kalki.",
            "Bhagavata Purana 7.8: Narasimha avatar manifested in Satya Yuga to protect Prahlada and slay the demon Hiranyakashipu.",
            "Bhagavata Purana 8.18–22: Vamana (dwarf avatar) approached the generous demon king Mahabali during a yajna and requested three paces of land, then expanded to cosmic form.",
            "Garuda Purana 1.86: Alternate lists sometimes place Balarama as the eighth avatar instead of Krishna.",
        ],
        answer_is_correct=False,
        factual_is_correct=False,
        structural_meta=StructuralMeta(
            required_terms=["Dashavatara", "Satya Yuga", "Mahabali", "Vishnu"],
            required_sections=["complete avatar list", "yuga assignment for key avatars", "purpose of each avatar", "evolutionary sequence note"],
            expected_order=["avatar list in order", "individual avatar details", "overarching pattern"],
            banned_terms=["Treta Yuga for Narasimha", "Ravana for Vamana"],
        ),
        structural_hints=[
            "List avatars in canonical order, then elaborate on key ones.",
            "Note the evolutionary progression from aquatic to human forms.",
            "Narasimha is Satya Yuga, not Treta.",
        ],
    ),
    Scenario(
        question="What is the story of Savitri and Satyavan from the Mahabharata?",
        given_answer="Savitri was a princess who chose Satyavan as her husband despite the sage Vishwamitra's warning that Satyavan would die within a year. When Yama came to claim Satyavan's soul, Savitri followed Yama for seven days. Impressed by her devotion, Yama granted three boons including Satyavan's life. The story is found in the Ramayana's Aranya Kanda.",
        ground_truth_answer="Savitri chose Satyavan despite the sage Narada's warning (not Vishwamitra) that Satyavan was fated to die within a year. When Yama came, Savitri followed him and through her wisdom and persistence obtained boons. The story appears in the Mahabharata's Vana Parva (not Ramayana's Aranya Kanda). Savitri's victory came through dialectical skill — she argued Yama into granting life, not through mere devotion alone.",
        ground_truth_citations=[
            "Mahabharata, Vana Parva, Chapters 277–283 (Pativrata Mahatmya)",
        ],
        available_passages=[
            "Mahabharata, Vana Parva 277: Sage Narada warned King Asvapati and Savitri that Satyavan, though virtuous, was destined to die exactly one year after their marriage.",
            "Mahabharata, Vana Parva 281–283: Savitri followed Yama as he carried Satyavan's soul. Through her persistent arguments and wisdom, Yama granted boons: her father-in-law's sight restored, his lost kingdom returned, and finally Satyavan's life.",
            "The Savitri-Satyavan story is entirely within the Mahabharata (Vana Parva). It does not appear in the Ramayana.",
        ],
        answer_is_correct=False,
        factual_is_correct=False,
        structural_meta=StructuralMeta(
            required_terms=["Narada", "Yama", "Vana Parva", "pativrata"],
            required_sections=["Savitri's choice", "Narada's warning", "confrontation with Yama", "method of victory", "source text"],
            expected_order=["Savitri's choice", "Narada's warning", "confrontation with Yama", "method of victory"],
            banned_terms=["Vishwamitra", "Aranya Kanda", "Ramayana"],
        ),
        structural_hints=[
            "Narrative structure: choice → prophecy → confrontation → resolution.",
            "Emphasise Savitri's dialectical skill, not just devotion.",
            "This story is in the Mahabharata, not the Ramayana.",
        ],
    ),
    Scenario(
        question="Describe the Samudra Manthan (Churning of the Ocean) episode.",
        given_answer="The devas and asuras churned the ocean of milk using Mount Meru as the churning rod and the serpent Takshaka as the rope. Vishnu took the form of Kurma (tortoise) to support the mountain. Fourteen treasures emerged including Amrita (nectar), Lakshmi, the deadly poison Halahala which was consumed by Vishnu, and the divine horse Uchchaihshravas. Mohini distributed the Amrita exclusively to the devas.",
        ground_truth_answer="The devas and asuras churned the ocean using Mount Mandara (not Meru) as the rod and the serpent Vasuki (not Takshaka) as the rope. Kurma supported Mandara on his back. Among the fourteen treasures were Amrita, Lakshmi, Halahala, and Uchchaihshravas. The Halahala poison was consumed by Shiva (not Vishnu), who held it in his throat, earning the name Neelakantha. Vishnu as Mohini distributed the Amrita.",
        ground_truth_citations=[
            "Bhagavata Purana 8.5–12",
            "Vishnu Purana 1.9",
            "Mahabharata, Adi Parva 15–17",
        ],
        available_passages=[
            "Bhagavata Purana 8.7: Mount Mandara was used as the churning rod. The serpent Vasuki served as the rope. Vishnu as Kurma supported the mountain from below.",
            "Bhagavata Purana 8.7: The Halahala poison emerged first, threatening all creation. Shiva drank the poison at Parvati's urging, holding it in his throat. His throat turned blue — hence the name Neelakantha.",
            "Vishnu Purana 1.9: Fourteen ratnas (treasures) emerged: Lakshmi, Kaustubha gem, Parijata tree, Varuni, Dhanvantari with Amrita, Chandra (moon), Kamadhenu, Airavata, Uchchaihshravas, and others.",
            "Bhagavata Purana 8.12: Vishnu assumed the form of Mohini to distribute Amrita among the devas, deceiving the asuras.",
        ],
        answer_is_correct=False,
        factual_is_correct=False,
        structural_meta=StructuralMeta(
            required_terms=["Mandara", "Vasuki", "Halahala", "Neelakantha", "Mohini", "ratna"],
            required_sections=["setup and participants", "churning mechanism", "crisis (poison)", "treasures", "resolution (Amrita distribution)"],
            expected_order=["setup and participants", "churning mechanism", "crisis (poison)", "treasures", "resolution (Amrita distribution)"],
            banned_terms=["Meru as churning rod", "Takshaka as rope", "Vishnu drank poison"],
        ),
        structural_hints=[
            "Follow the narrative arc: setup → churning → crisis → treasures → resolution.",
            "The mountain is Mandara, the serpent is Vasuki.",
            "Shiva consumed the poison, not Vishnu.",
        ],
    ),
    Scenario(
        question="What are the main philosophical schools (Darshanas) of Hinduism?",
        given_answer="The six orthodox darshanas are: Samkhya (founded by Kapila), Yoga (by Patanjali), Nyaya (by Gotama), Vaisheshika (by Kanada), Mimamsa (by Kumarila Bhatta), and Vedanta (by Shankaracharya). These are called astika schools because they accept the authority of the Vedas. The heterodox schools include Buddhism, Jainism, and the Charvaka school founded by Brihaspati.",
        ground_truth_answer="The six orthodox darshanas are: Samkhya (Kapila), Yoga (Patanjali), Nyaya (Gotama/Akshapada), Vaisheshika (Kanada), Purva Mimamsa (Jaimini, not Kumarila Bhatta — Kumarila was a later commentator), and Uttara Mimamsa/Vedanta (Badarayana/Vyasa composed the Brahma Sutras; Shankaracharya was a later commentator, not the founder). The heterodox (nastika) schools include Buddhism, Jainism, and Charvaka/Lokayata. Each school uses a distinct epistemological method (pramana).",
        ground_truth_citations=[
            "Sarva-Darshana-Sangraha by Madhvacharya",
            "Brahma Sutras by Badarayana",
            "Mimamsa Sutras by Jaimini",
        ],
        available_passages=[
            "The six astika darshanas accept Vedic authority. Purva Mimamsa was founded by Jaimini (author of Mimamsa Sutras). Kumarila Bhatta and Prabhakara were later commentators, not founders.",
            "Vedanta (Uttara Mimamsa) is based on the Brahma Sutras by Badarayana. Adi Shankaracharya (8th century CE) was the most influential commentator but not the founder of the school.",
            "Sarva-Darshana-Sangraha by Madhvacharya surveys all philosophical schools including both astika and nastika traditions.",
            "Charvaka (also called Lokayata) is a materialist school. Attribution to Brihaspati is traditional but historically uncertain.",
        ],
        answer_is_correct=False,
        factual_is_correct=False,
        structural_meta=StructuralMeta(
            required_terms=["astika", "nastika", "pramana", "Jaimini", "Badarayana"],
            required_sections=["astika schools with founders", "distinction between founder and commentator", "nastika schools", "epistemological note"],
            expected_order=["astika schools with founders", "nastika schools", "epistemological note"],
            banned_terms=["Kumarila Bhatta founded Mimamsa", "Shankaracharya founded Vedanta"],
        ),
        structural_hints=[
            "Distinguish between original founders and later commentators.",
            "Mention epistemological methods (pramanas) as a unifying thread.",
            "Mimamsa founder is Jaimini, Vedanta founder is Badarayana.",
        ],
    ),
    Scenario(
        question="Describe the concept of Rta in the Rigveda.",
        given_answer="Rta is the Rigvedic concept of cosmic order and truth that governs the universe. It is maintained by the god Indra, who is called Rtavan (possessor of Rta). The rivers flow, seasons change, and dawn appears because of Rta. Rta is the precursor to the later concept of Dharma. Varuna has no specific connection to Rta.",
        ground_truth_answer="Rta is the cosmic order governing natural and moral law in the Rigveda. Varuna (not Indra) is the principal guardian of Rta, called Rtasya Gopah (guardian of Rta). While Indra is important in the Rigveda, Rta's custodianship belongs to Varuna and to a lesser extent Mitra. Rta governs natural phenomena and ethical conduct, and is indeed the precursor to Dharma. The concept bridges cosmic regularity with human moral obligation.",
        ground_truth_citations=[
            "Rigveda 1.24.8",
            "Rigveda 7.87 (Varuna hymns)",
        ],
        available_passages=[
            "Rigveda 1.24.8: Varuna is praised as the upholder of Rta, the cosmic moral order.",
            "Rigveda 7.87: Hymns to Varuna describe him as Rtasya Gopah — the guardian of cosmic truth and order.",
            "Rta in the Rigveda encompasses both natural law (the regularity of cosmic phenomena) and moral law (truth and righteous conduct). It later evolved into the concept of Dharma.",
            "Indra is celebrated primarily as a warrior god (Vritrahan) and lord of storms, not as the guardian of Rta.",
        ],
        answer_is_correct=False,
        factual_is_correct=False,
        structural_meta=StructuralMeta(
            required_terms=["Rta", "Varuna", "Rtasya Gopah", "Dharma"],
            required_sections=["definition of Rta", "guardian deity", "natural law aspect", "moral law aspect", "evolution to Dharma"],
            expected_order=["definition of Rta", "guardian deity", "natural law aspect", "moral law aspect", "evolution to Dharma"],
            banned_terms=["Indra maintains Rta", "Varuna has no connection"],
        ),
        structural_hints=[
            "Distinguish natural-law and moral-law dimensions of Rta.",
            "Varuna is the guardian, not Indra.",
            "End with the Rta → Dharma evolution.",
        ],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Task registry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class TaskConfig:
    name: str
    description: str
    max_steps: int
    scenarios: List[Scenario]


TASKS: Dict[str, TaskConfig] = {
    "verify-factual": TaskConfig(
        name="verify-factual",
        description="Verify whether a given answer about Indic scriptures is factually correct and structurally sound. Accept correct answers, reject or edit incorrect ones.",
        max_steps=5,
        scenarios=VERIFY_FACTUAL_SCENARIOS,
    ),
    "correct-and-cite": TaskConfig(
        name="correct-and-cite",
        description="Improve a partially correct answer by fixing factual gaps, adding citations, and restructuring for coherence and proper terminology.",
        max_steps=8,
        scenarios=CORRECT_AND_CITE_SCENARIOS,
    ),
    "fix-hallucination": TaskConfig(
        name="fix-hallucination",
        description="Detect and correct subtle hallucinations, fix semantic structure, eliminate terminology errors, and ensure logical narrative flow.",
        max_steps=12,
        scenarios=FIX_HALLUCINATION_SCENARIOS,
    ),
}
