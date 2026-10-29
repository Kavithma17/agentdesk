import math
from types import SimpleNamespace

import pytest

from app.rag.embeddings import EmbeddingError, GeminiEmbedder, l2_normalize

DIM = 4


class FakeModels:
    """Stands in for `client.models`: records every request and returns made-up vectors.

    Each text "t<n>" gets the vector [n + 1, 1, 0, 0], so after normalizing,
    values[0] / values[1] == n + 1 tells us which text a vector belongs to.
    """

    def __init__(self, count_offset: int = 0, dim: int = DIM):
        self.calls: list[dict] = []
        self.count_offset = count_offset  # return this many extra (or fewer) vectors
        self.dim = dim

    def embed_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        texts = [content.parts[0].text for content in contents]
        vectors = [self._vector(text) for text in texts]
        vectors = vectors[: len(vectors) + self.count_offset] if self.count_offset < 0 else vectors
        vectors += [self._vector("t0")] * max(self.count_offset, 0)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in vectors])

    def _vector(self, text: str) -> list[float]:
        n = int(text[1:]) if text[1:].isdigit() else 0
        return [n + 1.0, 1.0] + [0.0] * (self.dim - 2)


def make_embedder(models: FakeModels, batch_size: int = 100) -> GeminiEmbedder:
    return GeminiEmbedder(SimpleNamespace(models=models), "test-model", DIM, batch_size)


def owner(vector: list[float]) -> int:
    """Which "t<n>" text a fake vector was made from."""
    return round(vector[0] / vector[1]) - 1


def test_returns_one_normalized_vector_per_text():
    vectors = make_embedder(FakeModels()).embed_documents(["t0", "t1", "t2"])

    assert len(vectors) == 3
    assert all(len(v) == DIM for v in vectors)
    assert all(math.isclose(math.hypot(*v), 1.0) for v in vectors)


def test_each_text_is_sent_in_its_own_content():
    # gemini-embedding-2 merges a plain list of strings into one vector.
    models = FakeModels()
    make_embedder(models).embed_documents(["t0", "t1"])

    contents = models.calls[0]["contents"]
    assert [content.parts[0].text for content in contents] == ["t0", "t1"]


def test_splits_into_batches_and_keeps_order():
    models = FakeModels()
    texts = [f"t{n}" for n in range(250)]

    vectors = make_embedder(models, batch_size=100).embed_documents(texts)

    assert [len(call["contents"]) for call in models.calls] == [100, 100, 50]
    assert [owner(v) for v in vectors] == list(range(250))


def test_documents_and_queries_use_different_task_types():
    models = FakeModels()
    embedder = make_embedder(models)

    embedder.embed_documents(["t0"])
    embedder.embed_query("t1")

    assert models.calls[0]["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert models.calls[1]["config"].task_type == "RETRIEVAL_QUERY"
    assert all(call["config"].output_dimensionality == DIM for call in models.calls)


def test_embed_query_returns_a_single_vector():
    vector = make_embedder(FakeModels()).embed_query("t7")

    assert len(vector) == DIM
    assert owner(vector) == 7


def test_no_texts_means_no_api_call():
    models = FakeModels()

    assert make_embedder(models).embed_documents([]) == []
    assert models.calls == []


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_rejects_blank_text_before_calling_the_api(blank):
    models = FakeModels()

    with pytest.raises(ValueError):
        make_embedder(models).embed_documents(["t0", blank])
    assert models.calls == []


@pytest.mark.parametrize("offset", [-1, 1])
def test_wrong_number_of_vectors_is_an_error(offset):
    with pytest.raises(EmbeddingError, match="texts but got"):
        make_embedder(FakeModels(count_offset=offset)).embed_documents(["t0", "t1"])


def test_wrong_dimension_is_an_error():
    with pytest.raises(EmbeddingError, match="dimensions"):
        make_embedder(FakeModels(dim=DIM + 1)).embed_documents(["t0"])


def test_name_includes_model_and_dimension():
    embedder = make_embedder(FakeModels())

    assert embedder.name == "gemini/test-model@4"
    assert embedder.dim == DIM


@pytest.mark.parametrize("batch_size", [0, 101])
def test_rejects_batch_size_outside_api_limits(batch_size):
    with pytest.raises(ValueError):
        make_embedder(FakeModels(), batch_size=batch_size)


def test_l2_normalize_gives_length_one():
    assert l2_normalize([3.0, 4.0]) == [0.6, 0.8]


def test_l2_normalize_rejects_zero_vector():
    with pytest.raises(EmbeddingError):
        l2_normalize([0.0, 0.0])
