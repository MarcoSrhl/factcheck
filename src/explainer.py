"""Explainability module for the fact-checking pipeline.

Provides multi-layered explanations for why a claim is judged as
SUPPORTED, REFUTED, or NOT ENOUGH INFO. Four complementary strategies
are combined by the master :class:`FactExplainer` class:

1. **KBReasoningExplainer** -- structured logical reasoning from KB evidence.
2. **T5ExplanationGenerator** -- free-form natural language explanation via T5.
3. **AttentionAnalyzer** -- BERT attention weight analysis.
4. **ConfidenceDecomposer** -- per-component confidence breakdown.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device selection helper (MPS-aware for Apple Silicon)
# ---------------------------------------------------------------------------


def _select_device(preferred: Optional[str] = None) -> torch.device:
    """Return the best available PyTorch device.

    Priority: *preferred* (if given) > MPS > CUDA > CPU.
    """
    if preferred:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =========================================================================
# 1. KBReasoningExplainer
# =========================================================================

# Mapping of common DBpedia ontology / property predicates to natural
# language descriptions.  Used to translate SPARQL predicate URIs into
# human-readable relation phrases.

PREDICATE_EXPLANATIONS: dict[str, str] = {
    # --- Ontology (dbo:) ---
    "http://dbpedia.org/ontology/capital": "is the capital of",
    "http://dbpedia.org/ontology/birthPlace": "was born in",
    "http://dbpedia.org/ontology/deathPlace": "died in",
    "http://dbpedia.org/ontology/country": "is in the country",
    "http://dbpedia.org/ontology/location": "is located in",
    "http://dbpedia.org/ontology/headquarter": "has its headquarters in",
    "http://dbpedia.org/ontology/foundationPlace": "was founded in",
    "http://dbpedia.org/ontology/nationality": "has the nationality",
    "http://dbpedia.org/ontology/spouse": "is the spouse of",
    "http://dbpedia.org/ontology/parent": "is the parent of",
    "http://dbpedia.org/ontology/child": "is the child of",
    "http://dbpedia.org/ontology/author": "was authored by",
    "http://dbpedia.org/ontology/director": "was directed by",
    "http://dbpedia.org/ontology/producer": "was produced by",
    "http://dbpedia.org/ontology/starring": "stars in",
    "http://dbpedia.org/ontology/team": "plays for the team",
    "http://dbpedia.org/ontology/league": "competes in the league",
    "http://dbpedia.org/ontology/occupation": "has the occupation",
    "http://dbpedia.org/ontology/almaMater": "studied at",
    "http://dbpedia.org/ontology/genre": "belongs to the genre",
    "http://dbpedia.org/ontology/language": "uses the language",
    "http://dbpedia.org/ontology/currency": "uses the currency",
    "http://dbpedia.org/ontology/populationTotal": "has a total population of",
    "http://dbpedia.org/ontology/areaTotal": "has a total area of",
    "http://dbpedia.org/ontology/elevation": "has an elevation of",
    "http://dbpedia.org/ontology/largestCity": "has as its largest city",
    "http://dbpedia.org/ontology/leader": "is led by",
    "http://dbpedia.org/ontology/leaderName": "has leader named",
    "http://dbpedia.org/ontology/president": "has as president",
    "http://dbpedia.org/ontology/party": "is a member of the party",
    "http://dbpedia.org/ontology/religion": "follows the religion",
    "http://dbpedia.org/ontology/knownFor": "is known for",
    "http://dbpedia.org/ontology/influenced": "influenced",
    "http://dbpedia.org/ontology/influencedBy": "was influenced by",
    "http://dbpedia.org/ontology/award": "received the award",
    "http://dbpedia.org/ontology/developer": "was developed by",
    "http://dbpedia.org/ontology/manufacturer": "is manufactured by",
    "http://dbpedia.org/ontology/founder": "was founded by",
    "http://dbpedia.org/ontology/owningCompany": "is owned by",
    "http://dbpedia.org/ontology/successor": "was succeeded by",
    "http://dbpedia.org/ontology/predecessor": "was preceded by",
    # --- Properties (dbp:) ---
    "http://dbpedia.org/property/capital": "is the capital of",
    "http://dbpedia.org/property/birthPlace": "was born in",
    "http://dbpedia.org/property/location": "is located in",
    "http://dbpedia.org/property/country": "is in the country",
    "http://dbpedia.org/property/nationality": "has nationality",
    "http://dbpedia.org/property/spouse": "is the spouse of",
    "http://dbpedia.org/property/author": "was authored by",
    # --- RDF / OWL ---
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type": "is of type",
    "http://www.w3.org/2002/07/owl#sameAs": "is the same entity as",
    "http://purl.org/dc/terms/subject": "has subject category",
}


def _predicate_to_human(predicate_uri: str) -> str:
    """Convert a predicate URI to a human-readable phrase.

    Falls back to extracting the local name from the URI if no mapping
    exists in :data:`PREDICATE_EXPLANATIONS`.
    """
    if predicate_uri in PREDICATE_EXPLANATIONS:
        return PREDICATE_EXPLANATIONS[predicate_uri]
    # Fallback: take the fragment after the last '/' or '#'
    local = predicate_uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    # camelCase -> space-separated lowercase
    readable = ""
    for ch in local:
        if ch.isupper() and readable and not readable.endswith(" "):
            readable += " "
        readable += ch.lower()
    return readable


class KBReasoningExplainer:
    """Build a structured reasoning chain from knowledge-base evidence.

    Given the KB evidence produced by the pipeline (found/not-found
    predicates between linked entities), this class constructs a
    step-by-step logical explanation in natural language.
    """

    def explain(self, fact_check_result: dict) -> dict[str, Any]:
        """Produce a reasoning chain from a fact-check result dict.

        Parameters
        ----------
        fact_check_result : dict
            The result dictionary returned by
            :meth:`FactChecker.check`.

        Returns
        -------
        dict
            Keys:
            - ``steps`` : list[str] -- ordered reasoning steps.
            - ``summary`` : str -- one-paragraph summary of the KB reasoning.
        """
        claim: str = fact_check_result.get("claim", "")
        triplets: list[tuple[str, str, str]] = fact_check_result.get("triplets", [])
        kb_evidence: list[dict] = fact_check_result.get("kb_evidence", [])
        entities: dict[str, str] = fact_check_result.get("entities", {})
        verdict: str = fact_check_result.get("verdict", "NOT ENOUGH INFO")

        steps: list[str] = []

        if not triplets:
            steps.append(
                f"Could not extract any (subject, predicate, object) "
                f"triplets from the claim: \"{claim}\"."
            )
            return {
                "steps": steps,
                "summary": (
                    "No structured information could be extracted from the "
                    "claim, so knowledge-base reasoning is not available."
                ),
            }

        for idx, evidence in enumerate(kb_evidence):
            triplet = evidence.get("triplet", ("?", "?", "?"))
            subj, pred, obj = triplet
            subj_uri = evidence.get("subject_uri")
            obj_uri = evidence.get("object_uri")
            found: bool = evidence.get("found", False)
            predicates: list[str] = evidence.get("predicates", [])

            # Step A: state what the claim asserts
            steps.append(
                f"Step {idx + 1}a: The claim states that "
                f"\"{subj}\" \"{pred}\" \"{obj}\"."
            )

            # Step B: entity linking result
            linked_parts: list[str] = []
            if subj_uri:
                linked_parts.append(f"\"{subj}\" -> <{subj_uri}>")
            else:
                linked_parts.append(
                    f"\"{subj}\" could not be linked to a DBpedia entity"
                )
            if obj_uri:
                linked_parts.append(f"\"{obj}\" -> <{obj_uri}>")
            else:
                linked_parts.append(
                    f"\"{obj}\" could not be linked to a DBpedia entity"
                )
            steps.append(
                f"Step {idx + 1}b: Entity linking results: "
                + "; ".join(linked_parts)
                + "."
            )

            # Step C: KB verification
            if found and predicates:
                human_preds = [_predicate_to_human(p) for p in predicates[:5]]
                pred_details = ", ".join(
                    f"\"{h}\" ({p.split('/')[-1]})"
                    for h, p in zip(human_preds, predicates[:5])
                )
                steps.append(
                    f"Step {idx + 1}c: DBpedia confirms a relation between "
                    f"\"{subj}\" and \"{obj}\" via: {pred_details}."
                )
                # Step D: semantic match assessment
                best_match = self._find_best_semantic_match(pred, predicates)
                if best_match:
                    human = _predicate_to_human(best_match)
                    local_name = best_match.split("/")[-1]
                    steps.append(
                        f"Step {idx + 1}d: The predicate <{local_name}> "
                        f"(\"{human}\") semantically matches the claim's "
                        f"relation \"{pred}\"."
                    )
                else:
                    steps.append(
                        f"Step {idx + 1}d: None of the found predicates "
                        f"closely match the claim's relation \"{pred}\", "
                        f"but a connection exists."
                    )
            elif not subj_uri or not obj_uri:
                missing = []
                if not subj_uri:
                    missing.append(f"\"{subj}\"")
                if not obj_uri:
                    missing.append(f"\"{obj}\"")
                steps.append(
                    f"Step {idx + 1}c: Could not query DBpedia because "
                    + " and ".join(missing)
                    + " could not be linked to known entities."
                )
            else:
                steps.append(
                    f"Step {idx + 1}c: No direct relation found between "
                    f"\"{subj}\" and \"{obj}\" in DBpedia."
                )

        # Build summary
        summary = self._build_summary(claim, kb_evidence, verdict)
        return {"steps": steps, "summary": summary}

    # ----- helpers -------------------------------------------------------

    @staticmethod
    def _find_best_semantic_match(
        claim_predicate: str, kb_predicates: list[str]
    ) -> Optional[str]:
        """Heuristic: find the KB predicate whose local name best
        overlaps with the words in the claim predicate."""
        claim_words = set(claim_predicate.lower().split())
        best: Optional[str] = None
        best_score = 0
        for p_uri in kb_predicates:
            local_name = p_uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
            # Split camelCase
            parts: list[str] = []
            current = ""
            for ch in local_name:
                if ch.isupper() and current:
                    parts.append(current.lower())
                    current = ch
                else:
                    current += ch
            if current:
                parts.append(current.lower())
            overlap = len(claim_words & set(parts))
            if overlap > best_score:
                best_score = overlap
                best = p_uri
        # Also consider the human-readable form from our dict
        if best_score == 0:
            for p_uri in kb_predicates:
                human = _predicate_to_human(p_uri).lower()
                human_words = set(human.split())
                overlap = len(claim_words & human_words)
                if overlap > best_score:
                    best_score = overlap
                    best = p_uri
        return best

    @staticmethod
    def _build_summary(
        claim: str,
        kb_evidence: list[dict],
        verdict: str,
    ) -> str:
        """Build a one-paragraph summary of KB reasoning."""
        total = len(kb_evidence)
        found_count = sum(1 for e in kb_evidence if e.get("found", False))

        if total == 0:
            return (
                f"No triplets could be verified against the knowledge base "
                f"for the claim \"{claim}\"."
            )

        if found_count == total:
            return (
                f"All {total} extracted relation(s) from the claim "
                f"\"{claim}\" were confirmed by DBpedia. "
                f"The knowledge base evidence supports a verdict of {verdict}."
            )
        elif found_count > 0:
            return (
                f"Out of {total} extracted relation(s), {found_count} "
                f"were confirmed by DBpedia for the claim \"{claim}\". "
                f"Partial evidence was found, leading to a verdict of {verdict}."
            )
        else:
            return (
                f"None of the {total} extracted relation(s) from the claim "
                f"\"{claim}\" could be confirmed by DBpedia. "
                f"The lack of supporting evidence contributes to a verdict "
                f"of {verdict}."
            )


# =========================================================================
# 2. T5ExplanationGenerator
# =========================================================================


class T5ExplanationGenerator:
    """Generate a natural-language explanation paragraph using T5.

    Uses ``t5-small`` by default. If a fine-tuned checkpoint exists at
    *model_path*, it is loaded instead. The generator is MPS-aware and
    will use Apple Metal acceleration on M-series Macs.

    Parameters
    ----------
    model_path : str or None
        Path to a fine-tuned T5 checkpoint directory. If ``None`` or the
        path does not exist, the base ``t5-small`` model is used.
    device : str or None
        Force a specific device (``"mps"``, ``"cuda"``, ``"cpu"``).
        Auto-detected when ``None``.
    """

    BASE_MODEL_NAME = "t5-small"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.device = _select_device(device)

        # Decide which checkpoint to load
        load_path = self.BASE_MODEL_NAME
        if model_path and os.path.isdir(model_path):
            load_path = model_path
            logger.info("Loading fine-tuned T5 from %s", model_path)
        else:
            if model_path:
                logger.warning(
                    "Fine-tuned T5 not found at '%s'. "
                    "Falling back to base %s.",
                    model_path,
                    self.BASE_MODEL_NAME,
                )
            logger.info("Using base T5 model: %s", self.BASE_MODEL_NAME)

        self.tokenizer = T5Tokenizer.from_pretrained(
            load_path, legacy=True
        )
        self.model = T5ForConditionalGeneration.from_pretrained(load_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info("T5 loaded on %s", self.device)

    def generate(
        self,
        claim: str,
        verdict: str,
        evidence: str,
        max_length: int = 150,
        num_beams: int = 4,
    ) -> str:
        """Generate a natural-language explanation.

        Parameters
        ----------
        claim : str
            The original claim text.
        verdict : str
            The verdict label (e.g., ``"SUPPORTED"``).
        evidence : str
            A textual summary of the evidence collected.
        max_length : int
            Maximum number of tokens in the generated explanation.
        num_beams : int
            Beam width for beam search decoding.

        Returns
        -------
        str
            A generated explanation paragraph.
        """
        input_text = (
            f"explain: claim: {claim} [SEP] verdict: {verdict} "
            f"[SEP] evidence: {evidence}"
        )

        input_ids = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
            )

        explanation = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )
        return explanation


# =========================================================================
# 3. AttentionAnalyzer
# =========================================================================


class AttentionAnalyzer:
    """Analyze BERT attention weights to identify the most influential tokens.

    Extracts attention from the final transformer layer and aggregates
    across all heads to determine which input tokens the model attends
    to most when making its prediction.
    """

    def analyze(
        self,
        claim: str,
        evidence: str,
        model: Any,
        tokenizer: Any,
        top_n: int = 10,
    ) -> dict[str, Any]:
        """Run attention analysis on a claim+evidence pair.

        Parameters
        ----------
        claim : str
            The claim text.
        evidence : str
            The evidence text.
        model : BertForSequenceClassification
            The BERT classifier model (must support
            ``output_attentions=True``).
        tokenizer : BertTokenizer
            The corresponding tokenizer.
        top_n : int
            Number of top-attended tokens to return.

        Returns
        -------
        dict
            - ``top_tokens`` : list[tuple[str, float]] -- (token, score).
            - ``attention_summary`` : str -- human-readable sentence.
            - ``all_token_scores`` : list[tuple[str, float]] -- every
              token with its aggregated attention score.
        """
        if model is None or tokenizer is None:
            return {
                "top_tokens": [],
                "attention_summary": (
                    "Attention analysis unavailable (no BERT model provided)."
                ),
                "all_token_scores": [],
            }

        text = f"{claim} [SEP] {evidence}" if evidence else claim

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # outputs.attentions is a tuple of (batch, heads, seq, seq) per layer
        attentions = outputs.attentions
        if not attentions:
            return {
                "top_tokens": [],
                "attention_summary": "No attention weights available.",
                "all_token_scores": [],
            }

        # Use the last layer's attention: (1, heads, seq_len, seq_len)
        last_layer_attention = attentions[-1]

        # Average across all heads: (1, seq_len, seq_len)
        avg_attention = last_layer_attention.mean(dim=1)

        # Sum attention received by each token (column-wise sum gives
        # how much other tokens attend TO each position)
        # Shape: (seq_len,)
        token_importance = avg_attention[0].sum(dim=0)

        # Normalise to [0, 1]
        total = token_importance.sum()
        if total > 0:
            token_importance = token_importance / total

        # Map back to tokens
        input_ids = inputs["input_ids"][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

        all_scores: list[tuple[str, float]] = []
        for tok, score in zip(tokens, token_importance.tolist()):
            # Skip special tokens for ranking purposes
            if tok in ("[PAD]", "[CLS]", "[SEP]"):
                continue
            all_scores.append((tok, round(score, 6)))

        # Sort descending by score
        all_scores.sort(key=lambda x: x[1], reverse=True)
        top_tokens = all_scores[:top_n]

        # Build human-readable summary
        if top_tokens:
            highlighted = ", ".join(
                f"'{tok}'" for tok, _ in top_tokens[:5]
            )
            attention_summary = (
                f"The model focused primarily on {highlighted} "
                f"when making its prediction."
            )
        else:
            attention_summary = "No significant token attention detected."

        return {
            "top_tokens": top_tokens,
            "attention_summary": attention_summary,
            "all_token_scores": all_scores,
        }


# =========================================================================
# 4. ConfidenceDecomposer
# =========================================================================


class ConfidenceDecomposer:
    """Break down the final verdict into per-component contributions.

    Shows how the knowledge base, neural classifier, and GAN
    discriminator each contributed to the combined verdict.
    """

    def decompose(self, fact_check_result: dict) -> dict[str, Any]:
        """Decompose a fact-check result into component contributions.

        Parameters
        ----------
        fact_check_result : dict
            The result dictionary from :meth:`FactChecker.check`.

        Returns
        -------
        dict
            - ``components`` : list[dict] -- per-component details.
            - ``combined_verdict`` : str
            - ``combined_confidence`` : float
            - ``formatted`` : str -- human-readable breakdown.
        """
        components: list[dict[str, Any]] = []

        # --- Knowledge Base -----------------------------------------------
        kb_evidence: list[dict] = fact_check_result.get("kb_evidence", [])
        kb_component = self._decompose_kb(kb_evidence)
        components.append(kb_component)

        # --- Neural Model -------------------------------------------------
        neural_pred: Optional[dict] = fact_check_result.get("neural_prediction")
        neural_component = self._decompose_neural(neural_pred)
        components.append(neural_component)

        # --- GAN Discriminator --------------------------------------------
        gan_score: Optional[float] = fact_check_result.get("gan_score")
        gan_component = self._decompose_gan(gan_score)
        components.append(gan_component)

        # --- Combined -----------------------------------------------------
        verdict = fact_check_result.get("verdict", "NOT ENOUGH INFO")
        confidence = fact_check_result.get("confidence", 0.0)

        formatted = self._format(components, verdict, confidence)

        return {
            "components": components,
            "combined_verdict": verdict,
            "combined_confidence": confidence,
            "formatted": formatted,
        }

    # ----- per-component helpers -----------------------------------------

    @staticmethod
    def _decompose_kb(kb_evidence: list[dict]) -> dict[str, Any]:
        """Summarise KB evidence contribution."""
        total = len(kb_evidence)
        found = sum(1 for e in kb_evidence if e.get("found", False))

        total_predicates = sum(
            len(e.get("predicates", [])) for e in kb_evidence if e.get("found")
        )

        if total == 0:
            status = "NO DATA"
            implication = "no information available"
        elif found == total:
            status = f"FOUND ({total_predicates} matching predicate(s))"
            implication = "supports SUPPORTED"
        elif found > 0:
            status = (
                f"PARTIAL ({found}/{total} relations found, "
                f"{total_predicates} predicate(s))"
            )
            implication = "weakly supports SUPPORTED"
        else:
            status = "NOT FOUND"
            implication = "supports REFUTED or NOT ENOUGH INFO"

        return {
            "name": "Knowledge Base",
            "status": status,
            "implication": implication,
            "details": {
                "total_triplets": total,
                "found_triplets": found,
                "total_predicates": total_predicates,
            },
        }

    @staticmethod
    def _decompose_neural(
        neural_pred: Optional[dict],
    ) -> dict[str, Any]:
        """Summarise neural model contribution."""
        if neural_pred is None:
            return {
                "name": "Neural Model",
                "status": "DISABLED",
                "implication": "not used in verdict",
                "details": {},
            }

        label = neural_pred.get("label", "UNKNOWN")
        conf = neural_pred.get("confidence", 0.0)
        probabilities = neural_pred.get("probabilities", {})

        return {
            "name": "Neural Model",
            "status": f"{label} with {conf:.2f} confidence",
            "implication": f"supports {label}",
            "details": {
                "label": label,
                "confidence": conf,
                "probabilities": probabilities,
            },
        }

    @staticmethod
    def _decompose_gan(gan_score: Optional[float]) -> dict[str, Any]:
        """Summarise GAN discriminator contribution."""
        if gan_score is None:
            return {
                "name": "GAN Discriminator",
                "status": "DISABLED",
                "implication": "not used in verdict",
                "details": {},
            }

        if gan_score >= 0.7:
            assessment = "triplet looks like a real fact"
        elif gan_score >= 0.4:
            assessment = "triplet is ambiguous"
        else:
            assessment = "triplet looks fabricated"

        return {
            "name": "GAN Discriminator",
            "status": f"{gan_score:.2f} ({assessment})",
            "implication": (
                "nudges confidence upward"
                if gan_score >= 0.5
                else "nudges confidence downward"
            ),
            "details": {
                "score": gan_score,
                "assessment": assessment,
            },
        }

    # ----- formatting ----------------------------------------------------

    @staticmethod
    def _format(
        components: list[dict[str, Any]],
        verdict: str,
        confidence: float,
    ) -> str:
        """Build a human-readable multi-line breakdown."""
        lines: list[str] = ["--- Confidence Breakdown ---"]
        for comp in components:
            name = comp["name"]
            status = comp["status"]
            impl = comp["implication"]
            lines.append(f"  {name}: {status} -> {impl}")
        lines.append(f"  Combined: {verdict} ({confidence:.2f})")
        return "\n".join(lines)


# =========================================================================
# 5. FactExplainer (master class)
# =========================================================================


class FactExplainer:
    """Master explainability class combining all four explanation strategies.

    Parameters
    ----------
    use_t5 : bool
        Whether to initialise the T5 explanation generator. Set to
        ``False`` to avoid the model download / memory overhead.
    t5_model_path : str or None
        Path to a fine-tuned T5 checkpoint. Falls back to ``t5-small``.
    use_attention : bool
        Whether to enable BERT attention analysis.
    device : str or None
        Force a device for T5 (MPS/CUDA/CPU). Auto-detected if ``None``.
    """

    def __init__(
        self,
        use_t5: bool = True,
        t5_model_path: Optional[str] = None,
        use_attention: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.kb_reasoner = KBReasoningExplainer()
        self.confidence_decomposer = ConfidenceDecomposer()

        self.t5_generator: Optional[T5ExplanationGenerator] = None
        if use_t5:
            try:
                self.t5_generator = T5ExplanationGenerator(
                    model_path=t5_model_path, device=device
                )
            except Exception as exc:
                logger.warning(
                    "Could not initialise T5 generator: %s. "
                    "Natural-language explanations will be unavailable.",
                    exc,
                )

        self.attention_analyzer: Optional[AttentionAnalyzer] = None
        if use_attention:
            self.attention_analyzer = AttentionAnalyzer()

    def explain(
        self,
        fact_check_result: dict,
        classifier: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Generate a complete multi-layered explanation.

        Parameters
        ----------
        fact_check_result : dict
            The result dict from :meth:`FactChecker.check`.
        classifier : FactClassifier or None
            If provided (and ``use_attention=True``), BERT attention
            weights are analysed. Pass the ``FactChecker.classifier``
            instance.

        Returns
        -------
        dict
            - ``reasoning_chain`` : dict from :class:`KBReasoningExplainer`.
            - ``natural_explanation`` : str from :class:`T5ExplanationGenerator`.
            - ``attention_analysis`` : dict from :class:`AttentionAnalyzer`.
            - ``confidence_breakdown`` : dict from :class:`ConfidenceDecomposer`.
            - ``summary`` : str -- one-paragraph combined summary.
        """
        # 1. KB reasoning
        reasoning_chain = self.kb_reasoner.explain(fact_check_result)

        # 2. T5 natural-language explanation
        natural_explanation: Optional[str] = None
        if self.t5_generator is not None:
            try:
                claim = fact_check_result.get("claim", "")
                verdict = fact_check_result.get("verdict", "NOT ENOUGH INFO")
                evidence_text = self._build_evidence_text(fact_check_result)
                natural_explanation = self.t5_generator.generate(
                    claim=claim,
                    verdict=verdict,
                    evidence=evidence_text,
                )
            except Exception as exc:
                logger.warning("T5 generation failed: %s", exc)
                natural_explanation = None

        # 3. Attention analysis
        attention_analysis: Optional[dict[str, Any]] = None
        if self.attention_analyzer is not None and classifier is not None:
            try:
                claim = fact_check_result.get("claim", "")
                evidence_text = self._build_evidence_text(fact_check_result)
                model = getattr(classifier, "model", None)
                tokenizer = getattr(classifier, "tokenizer", None)
                attention_analysis = self.attention_analyzer.analyze(
                    claim=claim,
                    evidence=evidence_text,
                    model=model,
                    tokenizer=tokenizer,
                )
            except Exception as exc:
                logger.warning("Attention analysis failed: %s", exc)
                attention_analysis = None

        # 4. Confidence breakdown
        confidence_breakdown = self.confidence_decomposer.decompose(
            fact_check_result
        )

        # 5. Build combined summary
        summary = self._build_combined_summary(
            fact_check_result=fact_check_result,
            reasoning_chain=reasoning_chain,
            natural_explanation=natural_explanation,
            attention_analysis=attention_analysis,
            confidence_breakdown=confidence_breakdown,
        )

        return {
            "reasoning_chain": reasoning_chain,
            "natural_explanation": natural_explanation,
            "attention_analysis": attention_analysis,
            "confidence_breakdown": confidence_breakdown,
            "summary": summary,
        }

    def format_explanation(self, explanation: dict) -> str:
        """Pretty-print a full explanation dict as a human-readable string.

        Parameters
        ----------
        explanation : dict
            The dict returned by :meth:`explain`.

        Returns
        -------
        str
            Multi-section formatted text.
        """
        sections: list[str] = []

        # --- Header -------------------------------------------------------
        sections.append("=" * 70)
        sections.append("FACT-CHECK EXPLANATION")
        sections.append("=" * 70)

        # --- KB Reasoning -------------------------------------------------
        reasoning = explanation.get("reasoning_chain")
        if reasoning:
            sections.append("")
            sections.append("--- Knowledge Base Reasoning ---")
            for step in reasoning.get("steps", []):
                sections.append(f"  {step}")
            summary = reasoning.get("summary", "")
            if summary:
                sections.append(f"\n  KB Summary: {summary}")

        # --- Natural Language Explanation ----------------------------------
        nl = explanation.get("natural_explanation")
        if nl:
            sections.append("")
            sections.append("--- Natural Language Explanation (T5) ---")
            sections.append(f"  {nl}")

        # --- Attention Analysis -------------------------------------------
        attention = explanation.get("attention_analysis")
        if attention:
            sections.append("")
            sections.append("--- Attention Analysis (BERT) ---")
            att_summary = attention.get("attention_summary", "")
            if att_summary:
                sections.append(f"  {att_summary}")
            top_tokens = attention.get("top_tokens", [])
            if top_tokens:
                token_strs = [
                    f"'{tok}' ({score:.4f})"
                    for tok, score in top_tokens[:10]
                ]
                sections.append(f"  Top tokens: {', '.join(token_strs)}")

        # --- Confidence Breakdown -----------------------------------------
        breakdown = explanation.get("confidence_breakdown")
        if breakdown:
            sections.append("")
            formatted = breakdown.get("formatted", "")
            if formatted:
                sections.append(formatted)

        # --- Summary ------------------------------------------------------
        summary = explanation.get("summary")
        if summary:
            sections.append("")
            sections.append("--- Overall Summary ---")
            sections.append(f"  {summary}")

        sections.append("")
        sections.append("=" * 70)
        return "\n".join(sections)

    # ----- internal helpers -----------------------------------------------

    @staticmethod
    def _build_evidence_text(fact_check_result: dict) -> str:
        """Reconstruct evidence text from KB results (same logic as
        FactChecker._build_evidence_text)."""
        kb_results = fact_check_result.get("kb_evidence", [])
        parts: list[str] = []
        for kr in kb_results:
            triplet = kr.get("triplet", ("?", "?", "?"))
            if kr.get("found"):
                preds = kr.get("predicates", [])[:3]
                pred_names = [p.split("/")[-1] for p in preds]
                parts.append(
                    f"{triplet[0]} is related to {triplet[2]} via "
                    + ", ".join(pred_names)
                )
            else:
                parts.append(
                    f"No relation found between {triplet[0]} and {triplet[2]}"
                )
        return ". ".join(parts) if parts else "No evidence available."

    @staticmethod
    def _build_combined_summary(
        fact_check_result: dict,
        reasoning_chain: dict,
        natural_explanation: Optional[str],
        attention_analysis: Optional[dict],
        confidence_breakdown: dict,
    ) -> str:
        """Build a single-paragraph summary combining all explanation layers."""
        claim = fact_check_result.get("claim", "")
        verdict = fact_check_result.get("verdict", "NOT ENOUGH INFO")
        confidence = fact_check_result.get("confidence", 0.0)

        parts: list[str] = []

        parts.append(
            f"The claim \"{claim}\" was judged as {verdict} "
            f"with {confidence:.0%} confidence."
        )

        # KB summary
        kb_summary = reasoning_chain.get("summary", "")
        if kb_summary:
            parts.append(kb_summary)

        # Attention highlight
        if attention_analysis:
            att_summary = attention_analysis.get("attention_summary", "")
            if att_summary and "unavailable" not in att_summary.lower():
                parts.append(att_summary)

        # Confidence components
        components = confidence_breakdown.get("components", [])
        active = [
            c for c in components if c.get("status") != "DISABLED"
        ]
        if active:
            comp_strs = [
                f"{c['name']}: {c['status']}" for c in active
            ]
            parts.append(
                "Component contributions: " + "; ".join(comp_strs) + "."
            )

        return " ".join(parts)


# =========================================================================
# 6. __main__ demonstration
# =========================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    # Ensure the project root is on the path so imports work
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.fact_checker import FactChecker, format_result

    print("Initialising FactChecker pipeline...")
    checker = FactChecker(use_neural=True, use_gan=False)

    print("Initialising FactExplainer (this may download T5 on first run)...")
    explainer = FactExplainer(
        use_t5=True,
        t5_model_path=None,
        use_attention=True,
    )

    demo_claims = [
        "Paris is the capital of France",
        "Barack Obama was born in Kenya",
        "The Eiffel Tower is located in Paris",
        "Albert Einstein developed the theory of relativity",
    ]

    for claim in demo_claims:
        print("\n" + "#" * 70)
        print(f"# CLAIM: {claim}")
        print("#" * 70)

        # Run the fact-checking pipeline
        result = checker.check(claim)
        print("\n" + format_result(result))

        # Generate explanation
        explanation = explainer.explain(
            fact_check_result=result,
            classifier=checker.classifier,
        )

        # Pretty-print
        print(explainer.format_explanation(explanation))
