# Automated Fact-Checking System Using DBpedia and Adversarial Learning

---

## Plan

1. **Introduction** — Contexte, problematique, objectif
2. **Presentation de la solution** — Architecture, pipeline, composants cles
3. **Resultats** — Metriques, evaluation, analyse des performances
4. **Conclusion** — Bilan, limites, perspectives

---

## 1. Introduction

### Contexte

Le fact-checking est la tache d'evaluer si une affirmation (claim) formulee en langage naturel est vraie ou fausse. Traditionnellement realisee par des professionnels, cette tache exige de croiser des sources factuelles, d'identifier les entites mentionnees, et de raisonner sur leurs relations.

Exemple : pour verifier *"Paris is the capital of France"*, un humain mobilise ses connaissances sur Paris, la France, et la relation "capitale". Un systeme automatise doit faire de meme, mais a partir d'une base de connaissances structuree.

### Problematique

Le fact-checking automatise reste un probleme ouvert. Les approches existantes se heurtent a plusieurs defis :
- **Extraction d'information** : transformer du texte libre en faits structures
- **Couverture des connaissances** : aucune base ne couvre tous les faits du monde
- **Ambiguite linguistique** : une meme affirmation peut etre formulee de multiples facons

### Objectif

Construire un systeme de fact-checking automatise qui, a partir d'une affirmation en langage naturel, determine si elle est **SUPPORTED** (vraie), **REFUTED** (fausse) ou **NOT ENOUGH INFO** (inverifiable), en utilisant **DBpedia** comme base de connaissances.

### Enjeux applicatifs

- Detection de desinformation dans les medias
- Filtrage de contenus sur les reseaux sociaux
- Construction de data lakes fiables a partir de donnees web

---

## 2. Presentation de la solution

### Architecture generale

Le systeme repose sur un pipeline multi-etapes dont le coeur est un **GAN (Generative Adversarial Network)** base sur BERT, entraine directement sur des faits extraits de DBpedia.

```
                         CLAIM (texte libre)
                               |
                               v
                    +---------------------+
                    | Triplet Extraction  |
                    |     (spaCy NLP)     |
                    +---------------------+
                               |
                     (sujet, relation, objet)
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
    +------------------------+   +------------------------+
    |    GAN Discriminator   |   |    Entity Linking      |
    |  (signal principal)    |   |   (DBpedia Lookup)     |
    |                        |   +------------------------+
    |  BERT encode le triplet|              |
    |  et predit real/fake   |              v
    +------------------------+   +------------------------+
                 |               |    SPARQL / JSON       |
                 |               |   (verification KB)    |
                 |               +------------------------+
                 |                           |
                 v                           v
            +------------------------------------+
            |        Combinaison des verdicts    |
            |   GAN (principal) + KB (support)   |
            +------------------------------------+
                            |
                            v
              SUPPORTED / REFUTED / NOT ENOUGH INFO
```

### Etape 1 : Extraction de triplets

A partir d'un claim en texte libre, le module `TripletExtractor` (base sur spaCy) extrait des triplets **(sujet, relation, objet)**.

| Claim | Triplet extrait |
|---|---|
| "Paris is the capital of France" | (Paris, capital, France) |
| "Barack Obama was born in Hawaii" | (Barack Obama, born in, Hawaii) |
| "Einstein developed the theory of relativity" | (Einstein, developed, theory of relativity) |

### Etape 2 : GAN Discriminator (coeur du systeme)

Le composant central est un **GAN adversarial** compose de :

**Generator (SwapGenerator)** : genere des triplets faux mais plausibles en remplacant le sujet ou l'objet par une entite du meme type (ex: remplacer "France" par "Germany" dans un triplet sur les capitales). Cela force le discriminateur a apprendre des distinctions *factuelles*, pas syntaxiques.

**Discriminator (BERTDiscriminator)** : un modele BERT (`bert-base-uncased`) avec une tete de classification binaire. Il apprend a distinguer les triplets reels (provenant de DBpedia) des triplets faux generes par le SwapGenerator.

```
Triplet: "Paris [REL] is capital of [REL] France"
                        |
                        v
                  BERT Encoder
                   (768-dim)
                        |
                        v
              Classification Head
           Linear(768→256) → LeakyReLU
           Linear(256→1)   → Sigmoid
                        |
                        v
                Score ∈ [0, 1]
          (0 = fake, 1 = real)
```

**Donnees d'entrainement** : les triplets sont fetches directement depuis DBpedia via des requetes SPARQL couvrant **41 categories** de relations (capitales, lieux de naissance, occupations, genres, nationalites, equipes sportives, etc.), avec **5000 triplets par categorie**.

| Categorie | Relation DBpedia | Exemple |
|---|---|---|
| capital_cities | dbo:capital | (Paris, is capital of, France) |
| birth_places | dbo:birthPlace | (Obama, was born in, Hawaii) |
| occupations | dbo:occupation | (Einstein, has occupation, Physicist) |
| genres | dbo:genre | (Inception, belongs to genre, Sci-Fi) |
| spouses | dbo:spouse | (Obama, is married to, Michelle Obama) |
| teams | dbo:team | (Messi, plays for, Inter Miami) |
| ... (41 categories) | ... | ... |

### Etape 3 : Entity Linking + Knowledge Base (signal secondaire)

En complement du GAN, le systeme tente de lier les entites extraites vers des URIs DBpedia pour une verification directe :

1. **Entity Linking** : recherche dans l'API DBpedia Lookup, avec un scoring multi-facteurs (similarite, popularite, qualite de l'URI)
2. **SPARQL Query** : verifie si une relation directe existe entre les entites dans le graphe DBpedia
3. **Property Check** : en cas d'absence de relation directe, examine les proprietes de l'entite pour detecter des contradictions

Ce signal KB ajuste la confidence du GAN mais ne le remplace pas.

### Etape 4 : Verdict final

Le score du GAN (0 a 1) est transforme en verdict :

| GAN Score | Verdict |
|---|---|
| > seuil haut | SUPPORTED |
| < seuil bas | REFUTED |
| entre les deux | NOT ENOUGH INFO |

La verification KB module la confidence : si les deux signaux s'accordent, la confidence augmente ; s'ils divergent, elle diminue.

### Stack technique

| Composant | Technologie |
|---|---|
| NLP / Triplet extraction | spaCy |
| Modele neural (GAN) | PyTorch + HuggingFace Transformers (BERT) |
| Knowledge Base | DBpedia (SPARQL + JSON API) |
| Tracking d'experiences | MLflow |
| Base de donnees | Neon (PostgreSQL) |

---

## 3. Resultats

### Performance du GAN Discriminator

Le GAN avec l'architecture entity-swap atteint **93% de precision** sur la tache de discrimination real/fake :

| Metrique | Valeur |
|---|---|
| Accuracy | 93% |
| Temps d'entrainement | ~8 minutes (20 epochs) |
| Batch size | 64 |
| Categories DBpedia | 41 |

A titre de comparaison, une architecture precedente (Gumbel-Softmax GAN) atteignait seulement 50% en 3.7 heures.

### Evaluation end-to-end du pipeline

Sur un jeu de validation de 49 claims :

| Label | Recall | Correct / Total |
|---|---|---|
| SUPPORTED | 40% | 8/20 |
| REFUTED | 55% | 11/20 |
| NOT ENOUGH INFO | 78% | 7/9 |
| **Global** | **53%** | **26/49** |

**Matrice de confusion :**

| | Pred. SUPPORTED | Pred. REFUTED | Pred. NEI |
|---|---|---|---|
| **SUPPORTED** | 8 | 5 | 7 |
| **REFUTED** | 5 | 11 | 4 |
| **NEI** | 0 | 2 | 7 |

### Analyse des erreurs

Les principales sources d'erreur identifiees :

1. **Entity Linking** : l'etape la plus fragile. Quand une entite est mal liee (ex: "Italy" vers "Palermo"), toute la verification KB echoue.
2. **Couverture DBpedia** : certains faits ne sont pas dans DBpedia, ce qui conduit a des faux NOT ENOUGH INFO.
3. **Extraction de triplets** : les phrases complexes ou ambigues donnent des triplets mal structures.

### Distribution de la confidence

| Verdict | Confidence moyenne |
|---|---|
| SUPPORTED | 0.70 |
| REFUTED | 0.75 |
| NOT ENOUGH INFO | 0.30 |

Le systeme est le plus confiant sur les claims REFUTED, et le moins confiant sur les claims NOT ENOUGH INFO, ce qui est coherent : l'absence d'information est inheremment plus difficile a affirmer.

---

## 4. Conclusion

### Bilan

Le systeme propose une approche originale du fact-checking automatise en combinant :
- Un **GAN adversarial a base de BERT** qui encode directement la connaissance factuelle de DBpedia
- Un pipeline de **verification par knowledge base** (entity linking + SPARQL) comme signal complementaire
- Une couverture de **41 categories de relations** issues de DBpedia

L'architecture GAN permet au systeme de fonctionner sans requete a DBpedia en temps reel a l'inference : la connaissance est internalisee dans les poids du discriminateur BERT.

### Limites

- **Performance globale (53%)** : encore insuffisante pour un deploiement en production
- **Entity linking** : maillon faible du pipeline, dont la fiabilite conditionne la qualite de la verification KB
- **Couverture** : bornee par le contenu de DBpedia, qui ne couvre pas tous les faits du monde
- **Triplets complexes** : les claims impliquant du raisonnement multi-etapes (negation, comparaisons, temporalite) restent hors de portee

### Perspectives

- **Augmenter les donnees d'entrainement** du GAN (41 categories x 5000 triplets) pour ameliorer la generalisation
- **Adapter le GAN en classifieur 3 classes** (SUPPORTED / REFUTED / NOT ENOUGH INFO) plutot que binaire, en utilisant le meme mecanisme d'entity-swap
- **Ameliorer l'entity linking** avec un scoring appris plutot que des poids fixes
- **Combiner avec d'autres bases de connaissances** (Wikidata, YAGO) pour augmenter la couverture
