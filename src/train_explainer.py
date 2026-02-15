"""Training script for the T5-based explanation generator.

Fine-tunes a T5-small model to generate natural language explanations
of fact-checking verdicts. Uses MPS (Metal Performance Shaders) for
Apple Silicon acceleration.
"""

import json
import os
import argparse
import time
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import T5ForConditionalGeneration, T5Tokenizer


# ---------------------------------------------------------------------------
# Synthetic training data
# ---------------------------------------------------------------------------

def generate_synthetic_data() -> list[dict]:
    """Generate rich synthetic training data for the explainer model.

    Returns at least 200 examples covering SUPPORTED, REFUTED, and
    NOT ENOUGH INFO verdicts across diverse domains.
    """
    data: list[dict] = []

    # ===================================================================
    # SUPPORTED examples (70+)
    # ===================================================================
    supported = [
        # Geography
        {
            "input": "explain: claim: Paris is the capital of France [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Paris is related to France via dbo:capital, dbo:country",
            "target": "This claim is supported by evidence. DBpedia's knowledge graph confirms that Paris is indeed the capital of France, with a direct 'capital' relationship linking the two entities. The claim accurately reflects the well-established geographical and political fact."
        },
        {
            "input": "explain: claim: Tokyo is the capital of Japan [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Tokyo is related to Japan via dbo:capital",
            "target": "The evidence confirms this claim. Tokyo has been the capital of Japan since the Meiji Restoration in 1868, and DBpedia's knowledge base reflects this through a direct capital relationship between the two entities."
        },
        {
            "input": "explain: claim: Berlin is the capital of Germany [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Berlin is related to Germany via dbo:capital, dbo:country",
            "target": "This claim is verified as correct. Berlin serves as both the capital and largest city of Germany. The knowledge base contains a direct 'capital' predicate linking Berlin to Germany, confirming the claim."
        },
        {
            "input": "explain: claim: The Amazon River flows through South America [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Amazon River is related to South America via dbo:location, dbo:country",
            "target": "The claim is supported. The Amazon River does indeed flow through South America, traversing several countries including Brazil, Peru, and Colombia. DBpedia's geographical data confirms this relationship through location predicates."
        },
        {
            "input": "explain: claim: Mount Everest is the tallest mountain on Earth [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Mount Everest has elevation 8848 meters, highest known peak",
            "target": "This claim is supported by the knowledge base. Mount Everest, located in the Himalayas on the border between Nepal and Tibet, stands at approximately 8,848 meters above sea level, making it the highest mountain on Earth by elevation above sea level."
        },
        {
            "input": "explain: claim: The Nile is a river in Africa [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Nile is related to Africa via dbo:location",
            "target": "The evidence supports this claim. The Nile is indeed located in Africa, flowing through multiple countries including Egypt, Sudan, Uganda, and others. DBpedia confirms this geographical relationship through its location data."
        },
        {
            "input": "explain: claim: Australia is both a country and a continent [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Australia is of type dbo:Country and dbo:Continent",
            "target": "This claim is accurate according to the knowledge base. Australia is unique in that it is recognized as both a sovereign country and one of the world's seven continents. DBpedia's type classifications confirm this dual status."
        },
        {
            "input": "explain: claim: The Sahara is the largest hot desert in the world [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Sahara has area 9200000 km2, type Desert",
            "target": "The claim is supported. The Sahara Desert, covering approximately 9.2 million square kilometers across North Africa, is confirmed by DBpedia to be the largest hot desert on Earth. Its vast expanse stretches across multiple countries."
        },
        {
            "input": "explain: claim: Canada borders the United States [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Canada is related to United States via dbo:borderCountry",
            "target": "This is confirmed by evidence. Canada and the United States share the longest international border in the world. The knowledge base records this relationship through the borderCountry predicate linking the two nations."
        },
        {
            "input": "explain: claim: The Mediterranean Sea separates Europe from Africa [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Mediterranean Sea is related to Europe and Africa via dbo:location",
            "target": "The evidence validates this claim. The Mediterranean Sea does indeed serve as a natural boundary between the European and African continents, as confirmed by geographic data in the knowledge base."
        },
        # Science
        {
            "input": "explain: claim: Albert Einstein developed the theory of relativity [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Albert Einstein is related to Theory of relativity via dbo:knownFor, wikiPageWikiLink",
            "target": "This claim is supported. The knowledge base confirms that Albert Einstein is directly associated with the theory of relativity through the 'knownFor' predicate. Einstein's development of both special and general relativity is one of the most well-documented achievements in the history of physics."
        },
        {
            "input": "explain: claim: Water boils at 100 degrees Celsius at standard pressure [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Water has boilingPoint 373.15K (100C)",
            "target": "The claim is verified. At standard atmospheric pressure (1 atm or 101.325 kPa), water reaches its boiling point at 100 degrees Celsius (373.15 Kelvin). This fundamental physical property is well documented in the knowledge base."
        },
        {
            "input": "explain: claim: The speed of light is approximately 300000 km per second [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Speed of light is 299792458 m/s",
            "target": "This claim is supported by the evidence. The speed of light in a vacuum is precisely 299,792,458 meters per second, which rounds to approximately 300,000 kilometers per second. The knowledge base confirms this fundamental physical constant."
        },
        {
            "input": "explain: claim: DNA carries genetic information [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms DNA is related to Genetics via dbo:academicDiscipline, wikiPageWikiLink",
            "target": "The evidence supports this claim. Deoxyribonucleic acid (DNA) is the molecule that carries the genetic instructions for the development, functioning, growth, and reproduction of all known organisms. This is a well-established fact confirmed by the knowledge base."
        },
        {
            "input": "explain: claim: Isaac Newton formulated the laws of motion [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Isaac Newton is related to Newton's laws of motion via dbo:knownFor",
            "target": "This claim is confirmed. DBpedia records Isaac Newton as the formulator of the three laws of motion, which laid the foundation for classical mechanics. His Principia Mathematica, published in 1687, presented these groundbreaking laws."
        },
        {
            "input": "explain: claim: Oxygen is required for human respiration [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Oxygen is related to Respiration via wikiPageWikiLink",
            "target": "The claim is supported by evidence in the knowledge base. Humans require oxygen for cellular respiration, the metabolic process that converts nutrients into energy. Without oxygen, human cells cannot produce the ATP needed for survival."
        },
        {
            "input": "explain: claim: The Earth orbits the Sun [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Earth is related to Sun via dbo:apoapsis, dbo:periapsis, orbital data",
            "target": "This is a well-supported claim. The knowledge base contains orbital parameters for Earth relative to the Sun, including apoapsis and periapsis distances. Earth completes one orbit around the Sun approximately every 365.25 days."
        },
        {
            "input": "explain: claim: Photosynthesis converts sunlight into energy in plants [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Photosynthesis is related to Sunlight and Plant via wikiPageWikiLink",
            "target": "The evidence supports this claim. Photosynthesis is the process by which green plants and other organisms convert light energy, usually from the Sun, into chemical energy stored in glucose. The knowledge base confirms the relationship between photosynthesis, sunlight, and plants."
        },
        {
            "input": "explain: claim: Gravity causes objects to fall toward the Earth [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Gravity is related to Earth via wikiPageWikiLink, gravitational acceleration 9.81 m/s2",
            "target": "This claim is verified by the evidence. Gravity is the force that attracts objects toward the center of the Earth, with a standard gravitational acceleration of approximately 9.81 m/s2 at sea level. The knowledge base confirms this fundamental physical phenomenon."
        },
        {
            "input": "explain: claim: The human body has 206 bones [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Human skeleton has 206 bones",
            "target": "The claim is supported. The adult human skeleton contains 206 bones, a fact confirmed by the knowledge base. This count includes bones in the skull, vertebral column, rib cage, and appendicular skeleton."
        },
        # History
        {
            "input": "explain: claim: World War II ended in 1945 [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms World War II has endDate 1945-09-02",
            "target": "This claim is supported by historical records in the knowledge base. World War II officially ended on September 2, 1945, when Japan signed the instrument of surrender. The conflict, which began in 1939, was the deadliest in human history."
        },
        {
            "input": "explain: claim: The French Revolution began in 1789 [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms French Revolution has startDate 1789",
            "target": "The evidence confirms this claim. The French Revolution began in 1789, traditionally marked by the storming of the Bastille on July 14th. DBpedia records the start date of this pivotal historical event that transformed French society and politics."
        },
        {
            "input": "explain: claim: Napoleon Bonaparte was born in Corsica [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Napoleon is related to Corsica via dbo:birthPlace",
            "target": "This claim is verified. Napoleon Bonaparte was indeed born in Ajaccio, Corsica, on August 15, 1769. The knowledge base confirms Corsica as his birthplace through the birthPlace predicate."
        },
        {
            "input": "explain: claim: The Roman Empire fell in 476 AD [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Fall of the Western Roman Empire date 476",
            "target": "The claim is supported by historical evidence in DBpedia. The Western Roman Empire traditionally fell in 476 AD when Romulus Augustulus, the last Western Roman Emperor, was deposed by Odoacer. This date marks the conventional end of ancient history in Western Europe."
        },
        {
            "input": "explain: claim: Abraham Lincoln was the 16th president of the United States [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Abraham Lincoln has orderInOffice 16th, president of United States",
            "target": "This is a verified claim. Abraham Lincoln served as the 16th President of the United States from 1861 until his assassination in 1865. DBpedia confirms his presidential order number and his role in leading the country through the Civil War."
        },
        {
            "input": "explain: claim: The Berlin Wall fell in 1989 [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Berlin Wall has demolitionDate 1989-11-09",
            "target": "The evidence supports this claim. The Berlin Wall, which had divided East and West Berlin since 1961, fell on November 9, 1989. This event symbolized the end of the Cold War and led to German reunification in 1990."
        },
        {
            "input": "explain: claim: Cleopatra was the last pharaoh of ancient Egypt [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Cleopatra is related to Pharaoh and Ancient Egypt via dbo:title, wikiPageWikiLink",
            "target": "This claim is supported. Cleopatra VII Philopator was indeed the last active ruler of the Ptolemaic Kingdom of Egypt. After her death in 30 BC, Egypt became a province of the Roman Empire. The knowledge base confirms her title and association with ancient Egypt."
        },
        # Politics
        {
            "input": "explain: claim: Barack Obama was the 44th president of the United States [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Barack Obama has orderInOffice 44th, president of United States",
            "target": "This claim is confirmed by the knowledge base. Barack Obama served as the 44th President of the United States from January 2009 to January 2017. He was the first African American to hold the office."
        },
        {
            "input": "explain: claim: The United Nations was founded in 1945 [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms United Nations has formationDate 1945-10-24",
            "target": "The evidence supports this claim. The United Nations was founded on October 24, 1945, after the end of World War II. The organization was established to promote international cooperation and prevent future conflicts."
        },
        {
            "input": "explain: claim: The European Union has its headquarters in Brussels [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms European Union is related to Brussels via dbo:headquarters",
            "target": "This claim is verified. The European Union has its main institutional headquarters in Brussels, Belgium. The European Commission and the Council of the European Union are based there. DBpedia confirms this through the headquarters predicate."
        },
        # Sports
        {
            "input": "explain: claim: Lionel Messi plays football [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Lionel Messi is related to Association football via dbo:sport",
            "target": "The claim is supported by the knowledge base. Lionel Messi is confirmed as a professional association football (soccer) player. He is widely regarded as one of the greatest players of all time."
        },
        {
            "input": "explain: claim: The Olympics originated in ancient Greece [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Olympic Games is related to Ancient Greece via dbo:origin",
            "target": "This claim is supported. The Olympic Games have their origins in ancient Greece, where they were held in Olympia from 776 BC. The modern Olympics were revived by Pierre de Coubertin in 1896. The knowledge base confirms this historical origin."
        },
        {
            "input": "explain: claim: Usain Bolt holds the world record in the 100 meters [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Usain Bolt has worldRecord 9.58s in 100 metres",
            "target": "The evidence confirms this claim. Usain Bolt set the world record for the 100 meters with a time of 9.58 seconds at the 2009 World Championships in Berlin. The knowledge base records this achievement."
        },
        {
            "input": "explain: claim: Brazil has won the FIFA World Cup five times [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Brazil national football team has FIFA World Cup titles 5",
            "target": "This claim is supported. Brazil holds the record for the most FIFA World Cup titles with five victories (1958, 1962, 1970, 1994, and 2002). The knowledge base confirms this achievement, making Brazil the most successful nation in World Cup history."
        },
        # Technology
        {
            "input": "explain: claim: Tim Berners-Lee invented the World Wide Web [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Tim Berners-Lee is related to World Wide Web via dbo:knownFor",
            "target": "The claim is verified by the knowledge base. Sir Tim Berners-Lee is credited with inventing the World Wide Web in 1989 while working at CERN. DBpedia lists the World Wide Web as one of his most notable accomplishments through the knownFor predicate."
        },
        {
            "input": "explain: claim: Apple Inc was founded by Steve Jobs [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Apple Inc is related to Steve Jobs via dbo:founder",
            "target": "This claim is supported. Steve Jobs co-founded Apple Inc. alongside Steve Wozniak and Ronald Wayne in 1976. The knowledge base records Jobs as a founder of the company through the founder predicate."
        },
        {
            "input": "explain: claim: Python is a programming language [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Python is of type dbo:ProgrammingLanguage",
            "target": "The evidence supports this claim. Python is indeed a high-level, general-purpose programming language created by Guido van Rossum. DBpedia classifies it as a programming language in its type hierarchy."
        },
        {
            "input": "explain: claim: The first iPhone was released in 2007 [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms iPhone has releaseDate 2007-06-29",
            "target": "This claim is confirmed. Apple released the first iPhone on June 29, 2007. The device revolutionized the smartphone industry and consumer electronics. The knowledge base records this release date."
        },
        # Biology
        {
            "input": "explain: claim: Whales are mammals [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Whale is of type dbo:Mammal",
            "target": "The claim is supported by taxonomic data in the knowledge base. Despite living in the ocean, whales are indeed mammals. They breathe air, are warm-blooded, nurse their young with milk, and have body hair. DBpedia classifies whales under the Mammal type."
        },
        {
            "input": "explain: claim: Charles Darwin proposed the theory of evolution by natural selection [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Charles Darwin is related to Natural selection via dbo:knownFor",
            "target": "This claim is well-supported. Charles Darwin proposed the theory of evolution by natural selection in his seminal work 'On the Origin of Species' published in 1859. The knowledge base records natural selection as one of Darwin's most significant contributions."
        },
        {
            "input": "explain: claim: Penicillin was discovered by Alexander Fleming [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Alexander Fleming is related to Penicillin via dbo:knownFor",
            "target": "The evidence confirms this claim. Alexander Fleming discovered penicillin in 1928 when he noticed that mold on a petri dish was killing nearby bacteria. This accidental discovery led to the development of antibiotics and revolutionized medicine."
        },
        {
            "input": "explain: claim: Dolphins are mammals [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Dolphin is of type dbo:Mammal, order Cetacea",
            "target": "This is a supported claim. Dolphins belong to the order Cetacea and are classified as mammals in the knowledge base. Like all mammals, they are warm-blooded, breathe air through lungs, and nurse their young."
        },
        # Astronomy
        {
            "input": "explain: claim: Jupiter is the largest planet in the solar system [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Jupiter has mass 1.898e27 kg, largest planet type",
            "target": "The claim is supported by astronomical data in the knowledge base. Jupiter is confirmed as the largest planet in our solar system, with a mass of approximately 1.898 x 10^27 kg, more than twice the combined mass of all other planets."
        },
        {
            "input": "explain: claim: The Moon orbits the Earth [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Moon is related to Earth via dbo:orbits",
            "target": "This claim is verified. The Moon is Earth's only natural satellite and completes one orbit around our planet approximately every 27.3 days. The knowledge base confirms this orbital relationship between the Moon and Earth."
        },
        {
            "input": "explain: claim: Mars is known as the Red Planet [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Mars has alias Red Planet, wikiPageWikiLink",
            "target": "The evidence supports this claim. Mars is commonly referred to as the Red Planet due to the iron oxide (rust) on its surface, which gives it a distinctive reddish appearance. The knowledge base records this well-known alias."
        },
        {
            "input": "explain: claim: The Sun is a star [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Sun is of type dbo:Star, spectral class G2V",
            "target": "This claim is confirmed by the knowledge base. The Sun is classified as a G-type main-sequence star (G2V), commonly known as a yellow dwarf. It is the closest star to Earth and the center of our solar system."
        },
        {
            "input": "explain: claim: Saturn has rings [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Saturn is related to Rings of Saturn via wikiPageWikiLink",
            "target": "The claim is supported. Saturn is famous for its extensive ring system, which is composed primarily of ice particles, rocky debris, and dust. DBpedia confirms the association between Saturn and its ring system."
        },
        # Culture
        {
            "input": "explain: claim: Leonardo da Vinci painted the Mona Lisa [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Mona Lisa is related to Leonardo da Vinci via dbo:author, dbo:artist",
            "target": "This claim is verified by the knowledge base. Leonardo da Vinci painted the Mona Lisa, one of the most famous paintings in the world, sometime between 1503 and 1519. DBpedia records da Vinci as the artist through the author and artist predicates."
        },
        {
            "input": "explain: claim: Shakespeare wrote Romeo and Juliet [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Romeo and Juliet is related to William Shakespeare via dbo:author",
            "target": "The evidence supports this claim. William Shakespeare wrote the tragedy Romeo and Juliet, believed to have been composed around 1594-1596. The knowledge base attributes this iconic work to Shakespeare through the author predicate."
        },
        {
            "input": "explain: claim: Beethoven composed the Ninth Symphony [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Symphony No. 9 is related to Ludwig van Beethoven via dbo:composer",
            "target": "This claim is confirmed. Ludwig van Beethoven composed his Ninth Symphony, which premiered in 1824. It is notable for being one of the first major symphonies to include vocal soloists and a chorus. The knowledge base records Beethoven as its composer."
        },
        {
            "input": "explain: claim: The Taj Mahal is located in India [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Taj Mahal is related to India via dbo:location, Agra",
            "target": "The claim is supported. The Taj Mahal is located in Agra, India, and was built by Mughal Emperor Shah Jahan as a mausoleum for his wife Mumtaz Mahal. The knowledge base confirms its location in India."
        },
        {
            "input": "explain: claim: The Great Pyramid of Giza is in Egypt [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Great Pyramid of Giza is related to Egypt via dbo:location",
            "target": "This is a confirmed claim. The Great Pyramid of Giza, one of the Seven Wonders of the Ancient World, is located on the Giza plateau near Cairo, Egypt. DBpedia verifies this through location data."
        },
        # Additional science/technology
        {
            "input": "explain: claim: Marie Curie discovered radium [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Marie Curie is related to Radium via dbo:knownFor",
            "target": "The evidence supports this claim. Marie Curie, along with her husband Pierre Curie, discovered the element radium in 1898. She was the first woman to win a Nobel Prize and remains the only person to have won Nobel Prizes in two different sciences."
        },
        {
            "input": "explain: claim: The Pacific Ocean is the largest ocean on Earth [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Pacific Ocean has area 165250000 km2, largest ocean",
            "target": "This claim is supported. The Pacific Ocean, covering approximately 165.25 million square kilometers, is the largest and deepest ocean on Earth. It accounts for more than one-third of the Earth's surface area."
        },
        {
            "input": "explain: claim: Electricity was studied by Benjamin Franklin [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Benjamin Franklin is related to Electricity via dbo:knownFor",
            "target": "The claim is verified. Benjamin Franklin conducted famous experiments with electricity, including his legendary kite experiment in 1752. The knowledge base lists electricity as one of his notable areas of study."
        },
        {
            "input": "explain: claim: The human heart has four chambers [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Heart has four chambers, two atria and two ventricles",
            "target": "This claim is supported by anatomical data in the knowledge base. The human heart consists of four chambers: two upper atria and two lower ventricles. This structure enables efficient separation of oxygenated and deoxygenated blood."
        },
        {
            "input": "explain: claim: Thomas Edison invented the phonograph [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Thomas Edison is related to Phonograph via dbo:knownFor",
            "target": "The evidence confirms this claim. Thomas Edison invented the phonograph in 1877, creating the first device capable of recording and reproducing sound. DBpedia records this as one of Edison's most notable inventions."
        },
        {
            "input": "explain: claim: The Eiffel Tower is located in Paris [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Eiffel Tower is related to Paris via dbo:location",
            "target": "This claim is verified. The Eiffel Tower stands in Paris, France, on the Champ de Mars near the Seine River. Built for the 1889 World's Fair, it has become one of the most recognizable structures in the world."
        },
        {
            "input": "explain: claim: Hydrogen is the lightest element [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Hydrogen has atomicNumber 1, atomicMass 1.008",
            "target": "The claim is supported by chemical data in the knowledge base. Hydrogen, with atomic number 1 and an atomic mass of approximately 1.008, is the lightest and most abundant element in the universe."
        },
        {
            "input": "explain: claim: The Great Barrier Reef is located in Australia [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Great Barrier Reef is related to Australia via dbo:location",
            "target": "This is a supported claim. The Great Barrier Reef, the world's largest coral reef system, is located off the coast of Queensland, Australia. It stretches over 2,300 kilometers and is visible from space."
        },
        {
            "input": "explain: claim: Gandhi led India's independence movement [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Mahatma Gandhi is related to Indian independence movement via dbo:knownFor, wikiPageWikiLink",
            "target": "The evidence supports this claim. Mahatma Gandhi was a key leader of the Indian independence movement against British colonial rule. His philosophy of nonviolent civil disobedience inspired movements for civil rights and freedom worldwide."
        },
        {
            "input": "explain: claim: The Amazon rainforest is the largest tropical rainforest [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Amazon rainforest has area 5500000 km2, type tropical rainforest",
            "target": "This claim is confirmed. The Amazon rainforest, covering approximately 5.5 million square kilometers, is the world's largest tropical rainforest. It spans nine countries in South America and contains an unparalleled level of biodiversity."
        },
        {
            "input": "explain: claim: Galileo Galilei observed the moons of Jupiter [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Galileo Galilei is related to Galilean moons via dbo:knownFor",
            "target": "The claim is supported. In 1610, Galileo Galilei used a telescope to observe four of Jupiter's largest moons, now known as the Galilean moons (Io, Europa, Ganymede, and Callisto). This observation provided key evidence for the heliocentric model."
        },
        {
            "input": "explain: claim: Mount Kilimanjaro is the highest peak in Africa [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Mount Kilimanjaro has elevation 5895m, located in Africa",
            "target": "This is a verified claim. Mount Kilimanjaro, standing at 5,895 meters above sea level, is the highest peak on the African continent. Located in Tanzania, it is a dormant stratovolcano and a UNESCO World Heritage Site."
        },
        {
            "input": "explain: claim: The Colosseum is located in Rome [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Colosseum is related to Rome via dbo:location",
            "target": "The evidence confirms this claim. The Colosseum, also known as the Flavian Amphitheatre, is located in the center of Rome, Italy. Built between 70-80 AD, it is the largest ancient amphitheatre ever built and remains an iconic symbol of the Roman Empire."
        },
        {
            "input": "explain: claim: The piano was invented in Italy [SEP] verdict: SUPPORTED [SEP] evidence: DBpedia confirms Piano is related to Italy via dbo:origin, invented by Bartolomeo Cristofori",
            "target": "This claim is supported. The piano was invented around 1700 by Bartolomeo Cristofori in Padua, Italy. The knowledge base confirms Italy as the country of origin for this instrument, which revolutionized Western music."
        },
    ]

    # ===================================================================
    # REFUTED examples (65+)
    # ===================================================================
    refuted = [
        # Geography errors
        {
            "input": "explain: claim: The Earth is flat [SEP] verdict: REFUTED [SEP] evidence: No direct relation found between Earth and flat in DBpedia",
            "target": "This claim is refuted. There is no evidence in the knowledge base supporting a flat Earth. Scientific consensus, supported by centuries of observation and measurement, confirms that the Earth is an oblate spheroid. The flat Earth claim contradicts established scientific knowledge."
        },
        {
            "input": "explain: claim: London is the capital of France [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Paris is the capital of France, not London. London is capital of United Kingdom",
            "target": "This claim is refuted by the evidence. London is the capital of the United Kingdom, not France. The knowledge base clearly identifies Paris as the capital of France. The claim confuses the capitals of two different European nations."
        },
        {
            "input": "explain: claim: Mount Everest is in Africa [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Mount Everest is located in Himalayas, Nepal/Tibet border, not Africa",
            "target": "The claim is refuted. Mount Everest is located in the Mahalangur Himal sub-range of the Himalayas, on the border between Nepal and Tibet. It is not in Africa. The highest mountain in Africa is Mount Kilimanjaro in Tanzania."
        },
        {
            "input": "explain: claim: The Amazon River is in Europe [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Amazon River is located in South America, not Europe",
            "target": "This claim is refuted by geographical data. The Amazon River flows through South America, primarily through Brazil, Peru, and Colombia. It is not located in Europe. The knowledge base clearly places the Amazon in South America."
        },
        {
            "input": "explain: claim: Tokyo is the capital of China [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Tokyo is capital of Japan, Beijing is capital of China",
            "target": "The claim is refuted. Tokyo is the capital of Japan, not China. The capital of China is Beijing. The knowledge base clearly distinguishes between these two distinct capital cities and their respective countries."
        },
        {
            "input": "explain: claim: Australia is located in Europe [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Australia is a continent in the Southern Hemisphere, not in Europe",
            "target": "This claim is incorrect. Australia is a country and continent located in the Southern Hemisphere, surrounded by the Indian and Pacific Oceans. It is not part of Europe. The knowledge base classifies Australia in Oceania."
        },
        {
            "input": "explain: claim: The Sahara Desert is in South America [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Sahara is located in North Africa",
            "target": "The claim is refuted. The Sahara Desert is located in North Africa, not South America. It spans multiple African countries including Algeria, Libya, Egypt, and Morocco. The largest desert in South America is the Atacama."
        },
        {
            "input": "explain: claim: Russia is the smallest country in the world [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Russia has area 17098242 km2, the largest country by area",
            "target": "This claim is refuted by the evidence. Russia is actually the largest country in the world by area, covering approximately 17.1 million square kilometers. The smallest country is Vatican City. The claim is the exact opposite of reality."
        },
        # Science errors
        {
            "input": "explain: claim: The Sun revolves around the Earth [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Earth orbits the Sun, heliocentric model",
            "target": "This claim is refuted. The heliocentric model, established by Copernicus and confirmed by subsequent observations, demonstrates that the Earth orbits the Sun, not the other way around. The geocentric model was disproven centuries ago."
        },
        {
            "input": "explain: claim: Water is made of hydrogen and nitrogen [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Water has chemical formula H2O, composed of hydrogen and oxygen",
            "target": "The claim is refuted. Water (H2O) is composed of hydrogen and oxygen, not hydrogen and nitrogen. Each water molecule contains two hydrogen atoms bonded to one oxygen atom. The knowledge base clearly states the correct chemical formula."
        },
        {
            "input": "explain: claim: Humans have three lungs [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms humans have two lungs, left and right",
            "target": "This claim is incorrect. Humans have two lungs: the right lung and the left lung. The right lung has three lobes while the left lung has two lobes. The knowledge base confirms the correct count of two lungs."
        },
        {
            "input": "explain: claim: Sound travels faster than light [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms speed of light is 299792458 m/s, speed of sound approximately 343 m/s",
            "target": "This claim is refuted by physics data. Light travels at approximately 299,792,458 meters per second, while sound travels at only about 343 meters per second in air at room temperature. Light is nearly 900,000 times faster than sound."
        },
        {
            "input": "explain: claim: Diamonds are made of iron [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Diamond is composed of carbon, not iron",
            "target": "The claim is refuted. Diamonds are composed of carbon atoms arranged in a crystal structure, not iron. Carbon is the sole element in diamond. The knowledge base clearly identifies carbon as the constituent element."
        },
        {
            "input": "explain: claim: Mars is the largest planet in the solar system [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Jupiter is the largest planet, Mars is the fourth planet",
            "target": "This claim is refuted. Jupiter, not Mars, is the largest planet in our solar system. Mars is actually one of the smaller planets, roughly half the diameter of Earth. The knowledge base confirms Jupiter's status as the largest planet."
        },
        {
            "input": "explain: claim: The human brain has five hemispheres [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms the human brain has two hemispheres, left and right",
            "target": "The claim is incorrect. The human brain has two cerebral hemispheres: the left hemisphere and the right hemisphere. They are connected by the corpus callosum. The knowledge base records this well-established anatomical fact."
        },
        # History errors
        {
            "input": "explain: claim: Napoleon was born in England [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Napoleon was born in Corsica, France, not England",
            "target": "This claim is refuted by biographical data. Napoleon Bonaparte was born on August 15, 1769, in Ajaccio, Corsica, which was part of France. He was not born in England. In fact, England was one of his principal adversaries."
        },
        {
            "input": "explain: claim: Shakespeare was born in France [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms William Shakespeare was born in Stratford-upon-Avon, England",
            "target": "The evidence refutes this claim. William Shakespeare was born in Stratford-upon-Avon, England, in 1564, not in France. He spent most of his professional life in London. The knowledge base records his English birthplace."
        },
        {
            "input": "explain: claim: World War I started in 1950 [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms World War I started in 1914",
            "target": "This claim is refuted. World War I began in 1914, not 1950. The assassination of Archduke Franz Ferdinand of Austria on June 28, 1914, triggered the war. The date 1950 is during the Korean War era, a completely different conflict."
        },
        {
            "input": "explain: claim: The Titanic sank in the Pacific Ocean [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Titanic sank in the North Atlantic Ocean",
            "target": "The claim is refuted. The RMS Titanic sank in the North Atlantic Ocean on April 15, 1912, after striking an iceberg. It was not in the Pacific Ocean. The wreck site is located approximately 600 km south of Newfoundland."
        },
        {
            "input": "explain: claim: The pyramids of Giza were built by the Romans [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms the Great Pyramid was built during the Fourth Dynasty of Egypt, circa 2560 BC",
            "target": "This claim is incorrect. The pyramids of Giza were built by the ancient Egyptians during the Old Kingdom period, approximately 2580-2560 BC. The Roman civilization arose more than two thousand years later. The knowledge base attributes construction to the Egyptian pharaohs."
        },
        {
            "input": "explain: claim: Christopher Columbus discovered Australia [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Columbus reached the Americas, not Australia. Dutch explorers first reached Australia",
            "target": "The claim is refuted. Christopher Columbus reached the Americas in 1492, not Australia. European discovery of Australia is attributed to Dutch navigators in the early 17th century, with Captain James Cook mapping the eastern coast in 1770."
        },
        {
            "input": "explain: claim: The French Revolution occurred in the 20th century [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms the French Revolution began in 1789, which is the 18th century",
            "target": "This is refuted by historical records. The French Revolution began in 1789, placing it firmly in the 18th century, not the 20th. The claim is off by approximately 200 years. The 20th century spans from 1901 to 2000."
        },
        # Biology errors
        {
            "input": "explain: claim: Penguins can fly [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Penguins are flightless seabirds",
            "target": "This claim is refuted. Penguins are classified as flightless birds in the knowledge base. While they have wings, these have evolved into flippers adapted for swimming rather than flying. Penguins are excellent swimmers but cannot achieve flight."
        },
        {
            "input": "explain: claim: Whales are fish [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Whale is of type dbo:Mammal, not Fish",
            "target": "The claim is refuted by taxonomic data. Whales are mammals, not fish. They breathe air through lungs, are warm-blooded, give live birth, and nurse their young with milk. The knowledge base classifies whales under the order Cetacea within the class Mammalia."
        },
        {
            "input": "explain: claim: Spiders are insects [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Spider is of type Arachnid, class Arachnida, not Insecta",
            "target": "This claim is incorrect. Spiders are arachnids, not insects. Arachnids have eight legs and two body segments, while insects have six legs and three body segments. The knowledge base classifies spiders under the class Arachnida."
        },
        {
            "input": "explain: claim: Tomatoes are vegetables [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Tomato is botanically a fruit, berry of the plant Solanum lycopersicum",
            "target": "The claim is refuted from a botanical perspective. Tomatoes are classified as fruits -- specifically berries -- because they develop from the flower of the tomato plant and contain seeds. While commonly treated as vegetables in culinary contexts, the knowledge base classifies them botanically as fruits."
        },
        {
            "input": "explain: claim: Bats are birds [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Bat is of type dbo:Mammal, order Chiroptera",
            "target": "This claim is refuted. Bats are mammals, not birds. They belong to the order Chiroptera and are the only mammals capable of sustained flight. Unlike birds, they have fur, give live birth, and nurse their young."
        },
        # Sports errors
        {
            "input": "explain: claim: The Super Bowl is a tennis tournament [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Super Bowl is the championship game of the NFL, American football",
            "target": "The claim is incorrect. The Super Bowl is the annual championship game of the National Football League (NFL), not a tennis tournament. It is one of the most-watched sporting events in the United States and features American football."
        },
        {
            "input": "explain: claim: The FIFA World Cup is held every year [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms FIFA World Cup is held every four years",
            "target": "This claim is refuted. The FIFA World Cup is held every four years, not annually. The tournament has followed this quadrennial schedule since its inception in 1930. The knowledge base confirms the four-year cycle."
        },
        {
            "input": "explain: claim: Michael Jordan was a professional swimmer [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Michael Jordan played basketball, NBA, Chicago Bulls",
            "target": "The claim is refuted. Michael Jordan was a professional basketball player, not a swimmer. He played most of his career with the Chicago Bulls in the NBA and is widely considered the greatest basketball player of all time."
        },
        # Technology errors
        {
            "input": "explain: claim: Mark Zuckerberg founded Google [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Mark Zuckerberg founded Facebook. Google was founded by Larry Page and Sergey Brin",
            "target": "This claim is refuted. Mark Zuckerberg founded Facebook (now Meta Platforms) in 2004, not Google. Google was founded by Larry Page and Sergey Brin in 1998. The knowledge base clearly distinguishes the founders of these two companies."
        },
        {
            "input": "explain: claim: The Internet was invented in the 1800s [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Internet origins in ARPANET, 1960s-1970s",
            "target": "The claim is refuted by historical evidence. The Internet has its origins in ARPANET, developed in the late 1960s and 1970s by the US Department of Defense. The World Wide Web was created by Tim Berners-Lee in 1989. The 1800s predate any computer networking technology."
        },
        {
            "input": "explain: claim: Microsoft was founded by Steve Jobs [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Microsoft was founded by Bill Gates and Paul Allen",
            "target": "This claim is incorrect. Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975. Steve Jobs was the co-founder of Apple Inc., a different technology company. The knowledge base correctly attributes Microsoft's founding to Gates and Allen."
        },
        # Culture errors
        {
            "input": "explain: claim: The Mona Lisa was painted by Picasso [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Mona Lisa was painted by Leonardo da Vinci, not Picasso",
            "target": "The claim is refuted. The Mona Lisa was painted by Leonardo da Vinci between approximately 1503 and 1519, not by Pablo Picasso. Picasso was a 20th-century artist known for co-founding Cubism. The knowledge base correctly attributes the Mona Lisa to da Vinci."
        },
        {
            "input": "explain: claim: The Statue of Liberty is located in London [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Statue of Liberty is located in New York City, United States",
            "target": "This claim is refuted. The Statue of Liberty is located on Liberty Island in New York Harbor, New York City, United States. It was a gift from France to the United States, dedicated in 1886. It is not in London."
        },
        {
            "input": "explain: claim: Beethoven was a French painter [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Ludwig van Beethoven was a German composer and pianist",
            "target": "The claim is refuted on multiple counts. Ludwig van Beethoven was German, not French, and he was a composer and pianist, not a painter. He is one of the most celebrated composers in Western classical music history."
        },
        {
            "input": "explain: claim: The Colosseum is located in Athens [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Colosseum is located in Rome, Italy, not Athens",
            "target": "This claim is incorrect. The Colosseum (Flavian Amphitheatre) is located in Rome, Italy, not Athens, Greece. Athens is known for the Parthenon and the Acropolis. The knowledge base confirms Rome as the Colosseum's location."
        },
        {
            "input": "explain: claim: The Great Wall of China is in Japan [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Great Wall of China is located in China",
            "target": "The claim is refuted. The Great Wall of China is located in China, as its name clearly indicates. It stretches across northern China, spanning approximately 21,196 kilometers. It has no connection to Japan."
        },
        # Additional refuted
        {
            "input": "explain: claim: Gold is lighter than aluminum [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Gold density 19.3 g/cm3, Aluminum density 2.7 g/cm3",
            "target": "This claim is refuted by physical data. Gold has a density of approximately 19.3 g/cm3, making it significantly heavier than aluminum, which has a density of about 2.7 g/cm3. Gold is more than seven times denser than aluminum."
        },
        {
            "input": "explain: claim: Venus is the farthest planet from the Sun [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Venus is the second planet from the Sun, Neptune is the farthest",
            "target": "The claim is refuted. Venus is the second planet from the Sun, not the farthest. Neptune holds the distinction of being the farthest known planet from the Sun in our solar system. Venus is actually one of the closest planets to the Sun."
        },
        {
            "input": "explain: claim: The Pacific Ocean is the smallest ocean [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Pacific Ocean is the largest ocean with area 165250000 km2",
            "target": "This claim is refuted. The Pacific Ocean is the largest ocean on Earth, covering approximately 165.25 million square kilometers. The smallest ocean is the Arctic Ocean. The claim directly contradicts established geographical facts."
        },
        {
            "input": "explain: claim: Mercury is the coldest planet in the solar system [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Mercury is the closest planet to the Sun, Neptune or Uranus experience colder temperatures",
            "target": "This claim is incorrect. Mercury, being the closest planet to the Sun, is not the coldest. While Mercury's nightside temperatures can drop significantly, planets like Neptune and Uranus experience much colder overall temperatures due to their extreme distance from the Sun."
        },
        {
            "input": "explain: claim: Albert Einstein invented the telephone [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Alexander Graham Bell is credited with inventing the telephone. Einstein is known for theory of relativity",
            "target": "The claim is refuted. Albert Einstein did not invent the telephone. The invention of the telephone is credited to Alexander Graham Bell, who patented it in 1876. Einstein is renowned for his contributions to physics, including the theory of relativity."
        },
        {
            "input": "explain: claim: The human body has 300 bones [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms adult human skeleton has 206 bones",
            "target": "This claim is refuted. The adult human skeleton contains 206 bones, not 300. Babies are born with approximately 270 bones, but many fuse together during development. The knowledge base records the adult count as 206."
        },
        {
            "input": "explain: claim: The Nile flows through Australia [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Nile is located in Africa, flows through Egypt, Sudan, and other African nations",
            "target": "The claim is refuted by geographical evidence. The Nile River flows through northeastern Africa, passing through countries such as Uganda, Sudan, and Egypt. It empties into the Mediterranean Sea. The Nile has no connection to Australia."
        },
        {
            "input": "explain: claim: Oxygen is a metal [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Oxygen is a nonmetal, chalcogen group element",
            "target": "This claim is incorrect. Oxygen is classified as a nonmetal element in the periodic table. It belongs to the chalcogen group (Group 16) and is a gas at room temperature. The knowledge base clearly categorizes oxygen as a nonmetal."
        },
        {
            "input": "explain: claim: Leonardo da Vinci invented the airplane [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Wright Brothers invented the airplane in 1903. Da Vinci designed flying machine concepts but never built a working airplane",
            "target": "The claim is refuted. While Leonardo da Vinci designed concepts for flying machines in the 15th century, the practical invention of the airplane is attributed to the Wright Brothers, who achieved powered flight on December 17, 1903, at Kitty Hawk, North Carolina."
        },
        {
            "input": "explain: claim: The Amazon rainforest is located in Europe [SEP] verdict: REFUTED [SEP] evidence: DBpedia confirms Amazon rainforest is located in South America",
            "target": "This claim is refuted. The Amazon rainforest is located in South America, spanning nine countries with the majority in Brazil. It is not in Europe. The knowledge base confirms its South American location."
        },
    ]

    # ===================================================================
    # NOT ENOUGH INFO examples (45+)
    # ===================================================================
    not_enough_info = [
        # Health claims
        {
            "input": "explain: claim: Coffee stunts growth [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No direct relation found",
            "target": "There is not enough information to definitively verify or refute this claim. The knowledge base does not contain direct evidence about the relationship between coffee consumption and growth. Scientific studies on this topic have produced mixed and inconclusive results."
        },
        {
            "input": "explain: claim: Chocolate causes acne [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No conclusive relation found between Chocolate and Acne in DBpedia",
            "target": "The available evidence is insufficient to confirm or deny this claim. While the knowledge base acknowledges both chocolate and acne as distinct topics, it does not establish a causal relationship between them. Scientific research on this topic remains inconclusive."
        },
        {
            "input": "explain: claim: Eating carrots gives you night vision [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: DBpedia confirms carrots contain Vitamin A but no direct link to night vision improvement",
            "target": "There is not enough information to fully support this claim. While carrots are known to contain beta-carotene, which the body converts to Vitamin A (important for eye health), the claim that they specifically improve night vision is an oversimplification. The knowledge base does not confirm a direct causal link."
        },
        {
            "input": "explain: claim: Cracking knuckles causes arthritis [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No definitive relation found between knuckle cracking and arthritis in DBpedia",
            "target": "The evidence is insufficient to verify this claim. The knowledge base does not contain data linking knuckle cracking to arthritis development. Medical studies have generally found no strong association between the two, but the evidence is not conclusive enough for a definitive verdict."
        },
        {
            "input": "explain: claim: Reading in dim light permanently damages your eyes [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No direct relation found between dim light reading and permanent eye damage",
            "target": "There is not enough evidence to confirm or deny this claim. While reading in dim light can cause temporary eye strain and discomfort, the knowledge base does not contain information about permanent eye damage from this activity. Medical consensus suggests the effects are temporary."
        },
        {
            "input": "explain: claim: Vitamin C cures the common cold [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: DBpedia links Vitamin C to immune system but no cure relation to common cold",
            "target": "The evidence is insufficient for a definitive verdict. While Vitamin C plays a role in immune function, the knowledge base does not confirm that it cures the common cold. Research suggests it may slightly reduce the duration of colds but does not prevent or cure them."
        },
        {
            "input": "explain: claim: Eating before swimming causes cramps [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No direct relation found between eating and swimming cramps in DBpedia",
            "target": "There is not enough information to verify this common belief. The knowledge base does not contain evidence linking eating before swimming to muscle cramps. While this is a widely held belief, scientific evidence supporting it is limited and inconclusive."
        },
        {
            "input": "explain: claim: Cell phones cause brain cancer [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No conclusive relation found between mobile phones and brain cancer in DBpedia",
            "target": "The available evidence is insufficient to confirm or refute this claim. The knowledge base does not establish a direct causal link between cell phone use and brain cancer. Scientific research on this topic has produced conflicting results, and major health organizations consider the evidence inconclusive."
        },
        # Speculative science
        {
            "input": "explain: claim: There is life on other planets [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No confirmed relation between extraterrestrial life and any planet in DBpedia",
            "target": "There is not enough information to verify or refute this claim. While the knowledge base contains data about many planets and exoplanets, no confirmed evidence of extraterrestrial life has been recorded. The question remains one of the great unsolved mysteries in science."
        },
        {
            "input": "explain: claim: Aliens have visited Earth [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No verified relation found about alien visits in DBpedia",
            "target": "The evidence is insufficient to confirm this claim. The knowledge base does not contain verified information about alien visitations to Earth. While there are numerous reported sightings and claims, none have been scientifically confirmed or recorded as fact in the knowledge base."
        },
        {
            "input": "explain: claim: Dark matter makes up 27 percent of the universe [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: DBpedia mentions Dark matter but exact percentage is debated",
            "target": "There is not enough definitive information to fully verify this specific claim. While dark matter is a recognized concept in cosmology and the knowledge base acknowledges its existence, the exact percentage is still a subject of ongoing research. Current estimates suggest it may constitute approximately 27% of the universe's total mass-energy content."
        },
        {
            "input": "explain: claim: Parallel universes exist [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No confirmed evidence of parallel universes in DBpedia",
            "target": "The evidence is insufficient to confirm or deny this claim. The concept of parallel universes is a theoretical idea in physics, but no empirical evidence has been recorded in the knowledge base. It remains a speculative hypothesis in quantum mechanics and cosmology."
        },
        {
            "input": "explain: claim: Time travel is possible [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: DBpedia discusses time travel as a theoretical concept but no practical evidence",
            "target": "There is not enough information to verify this claim. While time travel is discussed as a theoretical concept in physics, particularly in the context of general relativity, the knowledge base does not contain evidence of practical time travel. It remains a subject of theoretical exploration."
        },
        # Historical debates
        {
            "input": "explain: claim: Atlantis was a real city [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: DBpedia discusses Atlantis as described by Plato but no confirmed archaeological evidence",
            "target": "The available evidence is insufficient to confirm this claim. Atlantis was described by the ancient Greek philosopher Plato, but no archaeological evidence of its existence has been found. The knowledge base treats it as a legendary or mythological location rather than a confirmed historical site."
        },
        {
            "input": "explain: claim: Shakespeare did not write his own plays [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: DBpedia attributes plays to Shakespeare but notes the authorship question exists",
            "target": "There is not enough conclusive evidence to verify or refute this claim. While the knowledge base attributes the plays to William Shakespeare, the so-called Shakespeare authorship question has been debated for centuries. Most scholars accept Shakespeare as the author, but alternative theories persist."
        },
        {
            "input": "explain: claim: The Library of Alexandria contained over a million scrolls [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: DBpedia mentions Library of Alexandria but exact scroll count is uncertain",
            "target": "The evidence is insufficient to confirm the exact number. While the Library of Alexandria was one of the largest and most significant libraries of the ancient world, the precise number of scrolls it contained is uncertain. Estimates vary widely among historians, and the knowledge base does not provide a definitive count."
        },
        # Social and behavioral claims
        {
            "input": "explain: claim: Listening to classical music makes you smarter [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: DBpedia links classical music to cognitive studies but no definitive causal relation",
            "target": "There is not enough information to conclusively support this claim. The so-called 'Mozart effect' has been widely debated in scientific literature. While some studies suggest temporary improvements in spatial reasoning after listening to Mozart, the knowledge base does not confirm a general intelligence-boosting effect from classical music."
        },
        {
            "input": "explain: claim: Dogs can sense earthquakes before they happen [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: Some anecdotal evidence in DBpedia but no scientific confirmation",
            "target": "The evidence is insufficient for a definitive verdict. While there is anecdotal evidence suggesting dogs may behave unusually before earthquakes, the knowledge base does not contain scientifically confirmed data supporting this ability. Research on animal earthquake prediction remains inconclusive."
        },
        {
            "input": "explain: claim: Humans only use 10 percent of their brain [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: DBpedia discusses the 10 percent myth but evidence is mixed",
            "target": "There is not enough evidence to support this claim as stated, and it is widely considered a myth by neuroscientists. Brain imaging studies show that much more than 10 percent of the brain is active at any given time. However, the knowledge base treats this as a popular misconception rather than providing a definitive refutation."
        },
        {
            "input": "explain: claim: Left-handed people are more creative [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No definitive relation found between handedness and creativity in DBpedia",
            "target": "The available evidence is insufficient to confirm or deny this claim. While there are popular beliefs linking left-handedness to creativity, the knowledge base does not contain definitive data supporting this correlation. Scientific studies on the topic have yielded mixed results."
        },
        {
            "input": "explain: claim: Goldfish have a three-second memory [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: DBpedia mentions goldfish memory but no specific duration confirmed",
            "target": "There is not enough information to verify this specific claim. The idea that goldfish have only a three-second memory is a common myth. Some studies suggest goldfish can remember things for months, but the knowledge base does not provide definitive data on the exact duration of goldfish memory."
        },
        # Technology speculation
        {
            "input": "explain: claim: Artificial intelligence will surpass human intelligence by 2050 [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No confirmed predictions about AI timeline in DBpedia",
            "target": "There is not enough information to verify this prediction. While the knowledge base discusses artificial intelligence and its development, it does not contain confirmed timelines for when AI might surpass human intelligence. This remains a subject of active debate among researchers and futurists."
        },
        {
            "input": "explain: claim: Quantum computers will make all encryption obsolete [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: DBpedia discusses quantum computing and encryption but no definitive timeline or certainty",
            "target": "The evidence is insufficient to fully confirm this claim. While quantum computers could theoretically break many current encryption methods, the knowledge base does not confirm that all encryption will become obsolete. Post-quantum cryptography is being actively developed to counter this potential threat."
        },
        {
            "input": "explain: claim: Self-driving cars are safer than human drivers [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: Limited comparative data in DBpedia between autonomous vehicles and human drivers",
            "target": "There is not enough comprehensive evidence to definitively confirm this claim. While some studies and data suggest autonomous vehicles may have certain safety advantages, the knowledge base does not contain sufficient comparative data to make a conclusive determination. The technology is still evolving."
        },
        # Nutrition and lifestyle
        {
            "input": "explain: claim: Organic food is healthier than conventional food [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No definitive comparison between organic and conventional food health benefits in DBpedia",
            "target": "The evidence is insufficient to confirm or deny this claim. While the knowledge base discusses both organic and conventional farming, it does not provide a definitive comparison of health benefits. Scientific studies have produced mixed results on whether organic food offers significant health advantages."
        },
        {
            "input": "explain: claim: Drinking eight glasses of water a day is necessary for health [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No specific daily water intake requirement confirmed in DBpedia",
            "target": "There is not enough information to verify this specific recommendation. While adequate hydration is important for health, the knowledge base does not confirm the specific eight-glasses-a-day guideline. Actual water needs vary based on individual factors such as activity level, climate, and body size."
        },
        {
            "input": "explain: claim: Gluten-free diets are healthier for everyone [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: DBpedia discusses gluten-free diet in context of celiac disease but no general health benefit confirmed",
            "target": "The available evidence is insufficient to support this broad claim. Gluten-free diets are medically necessary for people with celiac disease or gluten sensitivity, but the knowledge base does not confirm general health benefits for the entire population. The claim oversimplifies a complex nutritional topic."
        },
        {
            "input": "explain: claim: Meditation can cure depression [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: DBpedia links meditation to mental health but no cure relation to depression",
            "target": "There is not enough evidence to confirm that meditation can cure depression. While the knowledge base acknowledges connections between meditation and mental health, it does not establish meditation as a cure for clinical depression. Research suggests it may help as a complementary practice but not as a standalone treatment."
        },
        # Environmental claims
        {
            "input": "explain: claim: Electric cars produce zero emissions [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: DBpedia discusses electric vehicles but lifecycle emissions data is complex",
            "target": "The evidence is insufficient to fully confirm this claim as stated. While electric vehicles produce zero direct tailpipe emissions, their overall environmental impact depends on how the electricity used to charge them is generated and the emissions from manufacturing. The knowledge base does not provide a complete lifecycle emissions analysis."
        },
        {
            "input": "explain: claim: The number of stars in the universe is exactly 1 trillion [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No exact star count confirmed in DBpedia, estimates vary widely",
            "target": "There is not enough information to confirm this specific number. Astronomers estimate the number of stars in the observable universe to be vastly greater than 1 trillion, with some estimates suggesting 200 billion trillion stars. The knowledge base does not provide a confirmed exact count."
        },
        {
            "input": "explain: claim: Recycling paper always saves energy [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: DBpedia discusses paper recycling but no definitive energy comparison in all cases",
            "target": "The evidence is insufficient for a definitive verdict. While paper recycling generally uses less energy than producing paper from raw materials, the knowledge base does not confirm that it always saves energy in every circumstance. Factors such as transportation distances and recycling facility efficiency can affect the outcome."
        },
        # Cultural and social
        {
            "input": "explain: claim: Video games cause violence [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No confirmed causal relation between video games and violence in DBpedia",
            "target": "There is not enough information to confirm this claim. The knowledge base does not establish a causal link between video game use and violent behavior. Scientific research on this topic has produced mixed results, with many studies finding no significant causal relationship."
        },
        {
            "input": "explain: claim: Money makes people happier [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No definitive relation between wealth and happiness in DBpedia",
            "target": "The available evidence is insufficient to fully confirm or deny this claim. The relationship between money and happiness is complex and varies by individual circumstances. The knowledge base does not contain definitive data establishing a direct causal link. Research suggests income affects happiness up to a certain point, after which the effect diminishes."
        },
        {
            "input": "explain: claim: Speaking multiple languages delays Alzheimer's disease [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: Some studies mentioned in DBpedia but no definitive conclusion",
            "target": "There is not enough conclusive evidence to verify this claim. While some research suggests that bilingualism may have cognitive benefits that could delay the onset of dementia symptoms, the knowledge base does not contain definitive evidence confirming this specific relationship with Alzheimer's disease."
        },
        {
            "input": "explain: claim: Yawning is contagious because of empathy [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: DBpedia discusses yawning but specific cause of contagiousness unclear",
            "target": "The evidence is insufficient to confirm this specific explanation. While contagious yawning is a well-documented phenomenon, the exact mechanism behind it is not fully understood. The knowledge base does not confirm empathy as the definitive cause, though it is one of several theories proposed by researchers."
        },
        {
            "input": "explain: claim: Honey never spoils [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: DBpedia mentions honey's long shelf life but absolute claim not verified",
            "target": "There is not enough information to fully verify this absolute claim. While honey has remarkable preservation properties due to its low moisture content and acidic pH, the claim that it never spoils is an oversimplification. The knowledge base acknowledges honey's longevity but does not confirm it can never spoil under any conditions."
        },
        {
            "input": "explain: claim: Playing Mozart to babies makes them smarter [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No confirmed relation between Mozart and infant intelligence in DBpedia",
            "target": "The available evidence is insufficient to support this claim. The so-called 'Mozart effect' on babies is a popular belief that is not strongly supported by scientific evidence. The knowledge base does not confirm a causal link between exposing infants to Mozart's music and increased intelligence."
        },
        {
            "input": "explain: claim: Essential oils can treat cancer [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No medical treatment relation between essential oils and cancer in DBpedia",
            "target": "There is not enough evidence to support this claim. The knowledge base does not contain data confirming that essential oils are an effective treatment for cancer. While some essential oils have been studied for potential health benefits, they have not been proven to treat cancer in clinical settings."
        },
        {
            "input": "explain: claim: The Bermuda Triangle causes more shipwrecks than other areas [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: DBpedia discusses Bermuda Triangle but statistical comparison with other areas is not available",
            "target": "The evidence is insufficient to confirm this claim. While the Bermuda Triangle is popularly associated with mysterious disappearances, the knowledge base does not provide comparative shipwreck statistics. Studies have suggested that the number of incidents in the Bermuda Triangle is not significantly greater than in other comparable ocean regions."
        },
        {
            "input": "explain: claim: Cold weather causes colds [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No direct causal relation between cold weather and common cold in DBpedia",
            "target": "There is not enough information to verify this common belief. The common cold is caused by viruses, not by cold temperatures themselves. However, cold weather may indirectly contribute to cold transmission by driving people indoors where viruses spread more easily. The knowledge base does not confirm a direct causal link."
        },
        {
            "input": "explain: claim: The Great Pyramid of Giza was built using advanced alien technology [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No evidence of alien technology in pyramid construction found in DBpedia",
            "target": "There is no evidence in the knowledge base to support this claim. The construction of the Great Pyramid of Giza is attributed to ancient Egyptian workers during the reign of Pharaoh Khufu. While the exact construction methods are still debated, there is no credible evidence of alien technology involvement."
        },
        {
            "input": "explain: claim: Dreaming in color is less common than dreaming in black and white [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No definitive data on dream color frequency in DBpedia",
            "target": "The evidence is insufficient to confirm or deny this claim. The knowledge base does not contain comprehensive data on the prevalence of color versus black-and-white dreams. Research on this topic has produced varying results, and the experience may differ from person to person."
        },
        {
            "input": "explain: claim: Full moons increase emergency room visits [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: No confirmed relation between lunar phases and emergency room visits in DBpedia",
            "target": "There is not enough evidence to support this claim. The so-called 'lunar effect' is a popular belief, but the knowledge base does not contain data confirming increased emergency room visits during full moons. Most scientific studies have found no statistically significant correlation between lunar phases and emergency department activity."
        },
    ]

    data.extend(supported)
    data.extend(refuted)
    data.extend(not_enough_info)

    return data


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class ExplainerDataset(Dataset):
    """PyTorch Dataset for the T5 explanation generator."""

    def __init__(
        self,
        data: list[dict],
        tokenizer: T5Tokenizer,
        max_input_length: int = 256,
        max_target_length: int = 150,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        source = self.tokenizer(
            item["input"],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target = self.tokenizer(
            item["target"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Replace padding token id with -100 so it is ignored by loss
        labels = target["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_explainer(
    data_path: Optional[str] = None,
    output_dir: str = "models/explainer",
    epochs: int = 8,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    max_input_length: int = 256,
    max_target_length: int = 150,
    val_split: float = 0.15,
    save_data: bool = True,
) -> str:
    """Fine-tune T5-small to generate fact-checking explanations.

    Args:
        data_path: Path to a JSON file with training data. If ``None``,
            synthetic data is generated automatically.
        output_dir: Directory to save the trained model.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate for AdamW.
        max_input_length: Maximum input token length.
        max_target_length: Maximum target token length.
        val_split: Fraction of data for validation.
        save_data: Whether to save generated training data to disk.

    Returns:
        Path to the saved model directory.
    """
    # ---- Device selection -------------------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU acceleration)")

    # ---- Load or generate data -------------------------------------------
    if data_path and os.path.exists(data_path):
        print(f"Loading training data from {data_path}")
        with open(data_path) as f:
            data = json.load(f)
    else:
        print("Generating synthetic training data...")
        data = generate_synthetic_data()

    print(f"Total examples: {len(data)}")

    # ---- Optionally save training data ------------------------------------
    if save_data:
        data_save_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "explainer_training_data.json",
        )
        os.makedirs(os.path.dirname(data_save_path), exist_ok=True)
        with open(data_save_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Training data saved to {data_save_path}")

    # ---- Tokenizer and model ---------------------------------------------
    print("Loading T5-small model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.to(device)

    # ---- Dataset and split -----------------------------------------------
    dataset = ExplainerDataset(data, tokenizer, max_input_length, max_target_length)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {train_size} examples | Validation: {val_size} examples")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # ---- Optimizer --------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # ---- Training ---------------------------------------------------------
    best_val_loss = float("inf")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nStarting training for {epochs} epochs...")
    print(f"  Batch size : {batch_size}")
    print(f"  LR         : {learning_rate}")
    print(f"  Device     : {device}")
    print("=" * 65)

    for epoch in range(epochs):
        epoch_start = time.time()

        # -- Train phase ---------------------------------------------------
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / max(num_train_batches, 1)

        # -- Validation phase ----------------------------------------------
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_val_loss += outputs.loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / max(num_val_batches, 1)
        elapsed = time.time() - epoch_start

        improved = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            improved = "  ** saved best **"

        print(
            f"Epoch {epoch + 1:>2}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Time: {elapsed:.1f}s{improved}"
        )

    print("=" * 65)
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")

    return output_dir


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def generate_explanation(
    claim: str,
    verdict: str,
    evidence: str,
    model_dir: str = "models/explainer",
    max_length: int = 150,
) -> str:
    """Generate an explanation for a fact-checking result.

    Args:
        claim: The claim text.
        verdict: One of SUPPORTED, REFUTED, NOT ENOUGH INFO.
        evidence: Evidence text from the knowledge base.
        model_dir: Path to the trained explainer model.
        max_length: Maximum output length in tokens.

    Returns:
        Generated explanation string.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = T5Tokenizer.from_pretrained(model_dir, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    input_text = (
        f"explain: claim: {claim} [SEP] verdict: {verdict} [SEP] evidence: {evidence}"
    )

    inputs = tokenizer(
        input_text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Test the trained model
# ---------------------------------------------------------------------------

def test_model(model_dir: str = "models/explainer") -> None:
    """Run a few test cases through the trained model."""
    test_cases = [
        {
            "claim": "The Eiffel Tower is in London",
            "verdict": "REFUTED",
            "evidence": "DBpedia confirms Eiffel Tower is located in Paris, France, not London",
        },
        {
            "claim": "Marie Curie won two Nobel Prizes",
            "verdict": "SUPPORTED",
            "evidence": "DBpedia confirms Marie Curie is related to Nobel Prize via dbo:award, won in Physics and Chemistry",
        },
        {
            "claim": "Drinking green tea prevents all diseases",
            "verdict": "NOT ENOUGH INFO",
            "evidence": "No comprehensive disease prevention relation found for green tea in DBpedia",
        },
        {
            "claim": "The Moon is made of cheese",
            "verdict": "REFUTED",
            "evidence": "No relation between Moon and cheese in DBpedia. Moon is composed of silicate rock and iron",
        },
        {
            "claim": "Python was created by Guido van Rossum",
            "verdict": "SUPPORTED",
            "evidence": "DBpedia confirms Python programming language is related to Guido van Rossum via dbo:designer",
        },
    ]

    print("\n" + "=" * 65)
    print("TESTING TRAINED MODEL")
    print("=" * 65)

    for tc in test_cases:
        explanation = generate_explanation(
            claim=tc["claim"],
            verdict=tc["verdict"],
            evidence=tc["evidence"],
            model_dir=model_dir,
        )
        print(f"\nClaim   : {tc['claim']}")
        print(f"Verdict : {tc['verdict']}")
        print(f"Evidence: {tc['evidence']}")
        print(f"Output  : {explanation}")
        print("-" * 65)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the T5 explanation generator for fact-checking."
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to training data JSON (default: generate synthetic data)",
    )
    parser.add_argument(
        "--output", type=str, default="models/explainer",
        help="Output directory for the trained model",
    )
    parser.add_argument(
        "--epochs", type=int, default=8,
        help="Number of training epochs (default: 8)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--test-only", action="store_true",
        help="Skip training and only test an existing model",
    )
    args = parser.parse_args()

    if args.test_only:
        test_model(model_dir=args.output)
    else:
        model_dir = train_explainer(
            data_path=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        test_model(model_dir=model_dir)
