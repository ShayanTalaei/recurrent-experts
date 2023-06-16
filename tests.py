import unittest
from typing import Callable

from torch import Tensor

from model import Expert, create_composition_with_indices, create_all_compositions
from transformer import (
    all_transformers,
    IdentityDetector,
    make_chains,
    Transformer,
    vertical_shift,
    horizontal_shift,
    rotate_left,
    horizontal_flip,
    Chain,
)


class ChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity_detector = IdentityDetector(123, shape=[3, 9], samples=5)

    def assertIsIdentity(self, t: Transformer | Callable[[Tensor], Tensor], msg: str = "") -> None:
        self.assertTrue(self.identity_detector.is_identity(t), msg=msg)

    def assertNotIdentity(self, t: Transformer | Callable[[Tensor], Tensor], msg: str = "") -> None:
        self.assertFalse(self.identity_detector.is_identity(t), msg=msg)

    def test_identity_property_of_forward_backward_transformations(self) -> None:
        for t in all_transformers:
            self.assertIsIdentity(
                lambda x: t.reverse(t.forward(x)),
                msg=f"backward(forward) failed for transformation: {t.get_name()}",
            )
            self.assertIsIdentity(
                lambda x: t.forward(t.reverse(x)),
                msg=f"forward(backward) failed for transformation: {t.get_name()}",
            )
            self.assertNotIdentity(lambda x: t.forward(x))
            self.assertNotIdentity(lambda x: t.reverse(x))

    def test_no_identities_in_make_chains_result(self) -> None:
        got_chains = make_chains(
            all_transformers,
            {1, 2, 3},
            avoid_repetition=True,
            identity_detector=IdentityDetector(123, shape=[3, 9], samples=5),
        )
        for chain in got_chains:
            self.assertNotIdentity(lambda x: chain.forward(x))
            self.assertNotIdentity(lambda x: chain.reverse(x))

    def test_all_unique_transformers_in_make_chains_result(self) -> None:
        expected_sizes = {1, 2, 3}
        got_chains = make_chains(
            all_transformers,
            expected_sizes,
            avoid_repetition=True,
        )

        for chain in got_chains:
            # expect to see only unique transformers inside a chain
            self.assertEqual(len(chain), len(set(chain)), f"got chain: {chain} with duplicate transformers")

        # expect to see all sizes of chain
        self.assertEqual(set(len(chain) for chain in got_chains), expected_sizes)

    def test_chain_name_and_abbreviation(self) -> None:
        chain = Chain([horizontal_flip, vertical_shift])
        self.assertEqual(chain.get_name(), "vertical_shift(horizontal_flip(x))")
        self.assertEqual(chain.get_abbreviation(), "vs(hf(x))")


class TransformerTests(unittest.TestCase):
    def test_vertical_shift(self) -> None:
        x = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        expected = Tensor([[5, 6], [7, 8], [1, 2], [3, 4]])
        got = vertical_shift.forward(x)
        self.assertTrue((got == expected).all())

    def test_horizontal_shift(self) -> None:
        x = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        expected = Tensor([[3, 4, 1, 2], [7, 8, 5, 6]])
        got = horizontal_shift.forward(x)
        self.assertTrue((got == expected).all())

    def test_rotate_left(self) -> None:
        x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected = Tensor([[3, 6, 9], [2, 5, 8], [1, 4, 7]])
        got = rotate_left.forward(x)
        self.assertTrue((got == expected).all())

    def test_horizontal_flip(self) -> None:
        x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected = Tensor([[3, 2, 1], [6, 5, 4], [9, 8, 7]])
        got = horizontal_flip.forward(x)
        self.assertTrue((got == expected).all())


class CompositionTests(unittest.TestCase):
    def test_create_composition_has_correct_names(self) -> None:
        experts = [Expert(None, None) for _ in range(3)]
        got_name = create_composition_with_indices(experts, list(range(len(experts)))).name
        expected_name = "e3(e2(e1(x)))"
        self.assertEqual(expected_name, got_name)

    def test_create_all_compositions_has_all_compositions(self) -> None:
        experts = [Expert(None, None) for _ in range(2)]
        got_names = set(comp.name for comp in create_all_compositions(experts, of_sizes={1, 2}, avoid_repetition=False))
        composition_names = {"e1(x)", "e2(x)", "e1(e1(x))", "e1(e2(x))", "e2(e1(x))", "e2(e2(x))"}
        self.assertEqual(composition_names, got_names)


if __name__ == "__main__":
    unittest.main()
