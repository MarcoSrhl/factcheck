"""Triplet extraction from English sentences using spaCy dependency parsing."""

import spacy


class TripletExtractor:
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = spacy.load(model_name)

    def extract(self, sentence: str) -> list[tuple[str, str, str]]:
        """Extract (subject, predicate, object) triplets from a sentence.

        Returns a list of (subject, predicate, object) tuples.
        """
        doc = self.nlp(sentence)
        triplets = []

        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "AUX":
                triplets.extend(self._extract_copular(token, doc))
            elif token.dep_ == "ROOT" and token.pos_ == "VERB":
                triplets.extend(self._extract_verbal(token, doc))

        if not triplets:
            triplets = self._fallback_extraction(doc)

        return triplets

    def _extract_copular(self, root, doc) -> list[tuple[str, str, str]]:
        """Handle copular constructions like 'Paris is the capital of France'."""
        triplets = []
        subject = None
        attribute = None

        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subject = self._get_span_text(child, doc)
            elif child.dep_ in ("attr", "acomp"):
                attribute = self._get_full_object(child, doc)

        if subject and attribute:
            triplets.append((subject, root.lemma_, attribute))

        return triplets

    def _extract_verbal(self, root, doc) -> list[tuple[str, str, str]]:
        """Handle verbal constructions like 'Obama was born in Hawaii'."""
        triplets = []
        subject = None
        verb_phrase = root.lemma_
        obj = None

        aux_parts = []
        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subject = self._get_span_text(child, doc)
            elif child.dep_ in ("dobj", "attr", "acomp"):
                obj = self._get_full_object(child, doc)
            elif child.dep_ in ("auxpass", "aux"):
                aux_parts.append(child.text)
            elif child.dep_ == "prep" and obj is None:
                prep_obj = self._get_prepositional_object(child)
                if prep_obj:
                    if aux_parts:
                        verb_phrase = " ".join(aux_parts) + " " + root.text + " " + child.text
                    else:
                        verb_phrase = root.lemma_ + " " + child.text
                    obj = prep_obj

        if subject and obj:
            triplets.append((subject, verb_phrase, obj))

        return triplets

    def _fallback_extraction(self, doc) -> list[tuple[str, str, str]]:
        """Fallback: find any subject-verb-object pattern."""
        triplets = []
        subjects = [t for t in doc if t.dep_ in ("nsubj", "nsubjpass")]
        for subj in subjects:
            verb = subj.head
            verb_text = verb.lemma_
            for child in verb.children:
                if child.dep_ in ("dobj", "attr", "acomp", "prep"):
                    if child.dep_ == "prep":
                        prep_obj = self._get_prepositional_object(child)
                        if prep_obj:
                            triplets.append((
                                self._get_span_text(subj, doc),
                                verb_text + " " + child.text,
                                prep_obj,
                            ))
                    else:
                        triplets.append((
                            self._get_span_text(subj, doc),
                            verb_text,
                            self._get_span_text(child, doc),
                        ))
        return triplets

    def _get_span_text(self, token, doc) -> str:
        """Get the full noun phrase containing this token."""
        for chunk in doc.noun_chunks:
            if token.i >= chunk.start and token.i < chunk.end:
                return chunk.text
        # If no noun chunk found, get compound + token
        compounds = [c.text for c in token.children if c.dep_ == "compound"]
        if compounds:
            return " ".join(compounds) + " " + token.text
        return token.text

    def _get_full_object(self, token, doc) -> str:
        """Get the full object phrase including prepositional complements."""
        base = self._get_span_text(token, doc)
        for child in token.children:
            if child.dep_ == "prep":
                prep_obj = self._get_prepositional_object(child)
                if prep_obj:
                    base += " " + child.text + " " + prep_obj
        return base

    def _get_prepositional_object(self, prep_token) -> str | None:
        """Extract the object of a prepositional phrase."""
        for child in prep_token.children:
            if child.dep_ == "pobj":
                compounds = [c.text for c in child.children if c.dep_ == "compound"]
                if compounds:
                    return " ".join(compounds) + " " + child.text
                return child.text
        return None


if __name__ == "__main__":
    extractor = TripletExtractor()

    test_sentences = [
        "Paris is the capital of France",
        "Barack Obama was born in Hawaii",
        "The Eiffel Tower is located in Paris",
        "Albert Einstein developed the theory of relativity",
    ]

    for sent in test_sentences:
        triplets = extractor.extract(sent)
        print(f"\nSentence: {sent}")
        for s, p, o in triplets:
            print(f"  -> ({s}, {p}, {o})")
