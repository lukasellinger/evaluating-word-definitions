"""Pipelines for the claim verification process."""


class Pipeline:
    """General Pipeline. Implement fetch_evidence, select_evidence, verify_claim."""

    def verify(self, word: str, claim: str) -> float:
        """
        Verify a claim related to a word. TODO: use numpy
        :param word: Word associated to the claim.
        :param claim: Claim to be verified.
        :return: percentage of true facts inside the claim.
        """
        ev_sents = self.fetch_evidence(word)
        atomic_claims = self.process_claim(claim)

        factuality = 0
        for atomic_claim in atomic_claims:
            selected_ev_sents = self.select_evidence(atomic_claim, ev_sents)
            factuality += self.verify_claim(atomic_claim, selected_ev_sents)

        return factuality / len(atomic_claims)

    @staticmethod
    def process_claim(claim: str) -> list[str]:
        """Process a claim. E.g. split it into its atomic facts."""
        return [claim]

    def fetch_evidence(self, word: str) -> list[str]:
        """
        Fetch the information of the word inside the knowledge base.
        :param word: Word, for which we need information.
        :return: List of sentences, representing all information known to the word.
        """

    def select_evidence(self, claim: str, sentences: list[str]) -> list[str]:
        """
        Select sentences possibly containing evidence for the claim.
        :param claim: Claim to be verified.
        :param sentences: Sentences to choose from.
        :return: List of sentences, possibly containing evidence.
        """

    def verify_claim(self, claim: str, sentences: list[str]) -> int:
        """
        Verify the claim using sentences as evidence.
        :param claim: Claim to be verified.
        :param sentences: Sentences to use as evidence.
        :return: 1, if claim can be verified, else 0.
        """


class WikiPipeline(Pipeline):
    """Pipeline using Wikipedia."""

    def __init__(self):
        super(WikiPipeline).__init__()

