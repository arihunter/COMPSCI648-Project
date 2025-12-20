from torch._refs import zero
import unittest
import torch
import math
from quantum_simulator import *

# Class with helper functions for tensor logic and unittest api
class TestHelper(unittest.TestCase):
    def assert_tensors_close(self, actual, expected):
        try:
            torch.testing.assert_close(actual, expected)
        except AssertionError as e:
            self.fail(f"\nExpected: {expected}\nReceived: {actual}\nOriginal error: {e}")

class TestStates(TestHelper):
    def test_zero_state(self):
        state = zero_state(1)
        self.assertEqual(state[0], torch.complex(torch.tensor(1.0), torch.tensor(0.0)))
        self.assertEqual(state[1], torch.complex(torch.tensor(0.0), torch.tensor(0.0)))

    def test_random_state(self):
        state = random_state(2)
        self.assertAlmostEqual(torch.linalg.norm(state).item(), 1.0, places=5)

    def test_custom_state(self):
        amplitudes = [1, 0, 0, 1j]
        state = custom_state(amplitudes)
        norm = torch.linalg.norm(state)
        self.assertAlmostEqual(norm.item(), 1.0, places=5)

    def test_measure(self):
        state = zero_state(1)
        counts = measure(state, shots=1000)
        self.assertIn('0', counts)
        self.assertNotIn('1', counts)

        state = apply_gate(state, X, [0], 1)
        counts = measure(state, shots=1000)
        self.assertIn('1', counts)
        self.assertNotIn('0', counts)
        
    def test_3q_state(self):
        state = zero_state(3)
        expected = torch.zeros(8, dtype=torch.cfloat)
        expected[0] = 1  # |000>
        self.assert_tensors_close(state, expected)

class TestGates(TestHelper):
    def test_apply_gate(self): # Identity
        state = zero_state(1)
        new_state = apply_gate(state, I2, [0], 1)
        expected = torch.tensor([1, 0], dtype=torch.cfloat)
        self.assert_tensors_close(new_state, expected)

    def test_X_gate(self):
        state = zero_state(1)
        new_state = apply_gate(state, X, [0], 1)
        expected = torch.tensor([0, 1], dtype=torch.cfloat)
        self.assert_tensors_close(new_state, expected)

    # y gate application must be wrong
    def test_Y_gate(self):
        state = zero_state(1);  # |0>,
        new_state = apply_gate(state, Y, [0], 1) # Y|0> = i|1>
        expected = torch.tensor([0, 1j], dtype=torch.cfloat)
        self.assert_tensors_close(new_state, expected)
        
        state = zero_state(1); state[0] = 0; state[1] = 1;  # |1>,
        new_state = apply_gate(state, Y, [0], 1) # Y|1> = -i|0>
        expected = torch.tensor([-1j, 0], dtype=torch.cfloat)
        self.assert_tensors_close(new_state, expected)

    def test_Z_gate(self):
        state = zero_state(1)
        state[0] = 0;state[1] = 1  # |1>
        new_state = apply_gate(state, Z, [0], 1)
        expected = torch.tensor([0, -1], dtype=torch.cfloat)
        self.assert_tensors_close(new_state, expected)

    def test_H_gate(self):
        state = zero_state(1) # |0>
        new_state = apply_gate(state, H, [0], 1)
        expected = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)], dtype=torch.cfloat)
        self.assert_tensors_close(new_state, expected)

    def test_S_gate(self):
        state = zero_state(1)
        state[0] = 0;state[1] = 1  # |1>
        new_state = apply_gate(state, S, [0], 1)
        expected = torch.tensor([0, 1j], dtype=torch.cfloat)
        self.assert_tensors_close(new_state, expected)

    def test_T_gate(self):
        state = zero_state(1)
        state[0] = 0;state[1] = 1  # |1>
        new_state = apply_gate(state, T, [0], 1)
        phase = torch.exp(torch.complex(torch.tensor(0.0), torch.tensor(math.pi/4)))
        expected = torch.tensor([0, phase], dtype=torch.cfloat)
        self.assert_tensors_close(new_state, expected)

    def test_RX_gate(self):
        theta = math.pi
        gate = RX(theta)
        state = zero_state(1)
        new_state = apply_gate(state, gate, [0], 1)
        expected = torch.tensor([0, -1j], dtype=torch.cfloat)  # RX(pi) |0> = -i |1>
        self.assert_tensors_close(new_state, expected)

    def test_RY_gate(self):
        theta = math.pi
        gate = RY(theta)
        state = zero_state(1)
        new_state = apply_gate(state, gate, [0], 1)
        expected = torch.tensor([0, 1], dtype=torch.cfloat)  # RY(pi) |0> = |1>
        self.assert_tensors_close(new_state, expected)

    def test_RZ_gate(self):
        theta = math.pi
        gate = RZ(theta)
        state = zero_state(1)
        state[0] = 0;state[1] = 1  # |1>
        new_state = apply_gate(state, gate, [0], 1)
        expected = torch.tensor([0, 1j], dtype=torch.cfloat)  # RZ(pi) |1> = i |1>
        self.assert_tensors_close(new_state, expected)

    def test_CNOT_gate_00(self):
        # CNOT |00> = |00>
        state = zero_state(2)
        new_state = apply_gate(state, CNOT, [0, 1], 2)
        expected = zero_state(2)
        self.assert_tensors_close(new_state, expected)

    def test_CNOT_gate_01(self):
        # CNOT |01> = |01>
        state = zero_state(2)
        state[0] = 0;state[1] = 1
        new_state = apply_gate(state, CNOT, [0, 1], 2)
        expected = torch.zeros(4, dtype=torch.cfloat)
        expected[1] = 1
        self.assert_tensors_close(new_state, expected)

    def test_CNOT_gate_10(self):
        # CNOT |10> = |11>
        state = zero_state(2)
        state[0] = 0; state[2] = 1
        new_state = apply_gate(state, CNOT, [0, 1], 2)
        expected = torch.zeros(4, dtype=torch.cfloat)
        expected[3] = 1
        self.assert_tensors_close(new_state, expected)

    def test_CNOT_gate_11(self):
        # CNOT |11> = |10>
        state = zero_state(2)
        state[0] = 0;state[3] = 1
        new_state = apply_gate(state, CNOT, [0, 1], 2)
        expected = torch.zeros(4, dtype=torch.cfloat)
        expected[2] = 1
        self.assert_tensors_close(new_state, expected)

    def test_CNOT_gate_plus_plus(self):
        # CNOT |++> = |++>
        plus = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)], dtype=torch.cfloat)
        state = torch.kron(plus, plus)
        new_state = apply_gate(state, CNOT, [0, 1], 2)
        expected = state
        self.assert_tensors_close(new_state, expected)

    def test_CNOT_gate_plus_minus(self):
        # CNOT |+-> = |-->
        plus = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)], dtype=torch.cfloat)
        minus = torch.tensor([1/math.sqrt(2), -1/math.sqrt(2)], dtype=torch.cfloat)
        state = torch.kron(plus, minus)
        new_state = apply_gate(state, CNOT, [0, 1], 2)
        expected = torch.kron(minus, minus)
        self.assert_tensors_close(new_state, expected)

    def test_X_Y_Z_anticommute(self):
        state = random_state(1)
        # X and Y anticommute: {X,Y} = 0
        xy = apply_gate(apply_gate(state, X, [0], 1), Y, [0], 1)
        yx = apply_gate(apply_gate(state, Y, [0], 1), X, [0], 1)
        anticommutator = xy + yx
        self.assert_tensors_close(anticommutator, torch.zeros_like(anticommutator))

        # X and Z anticommute: {X,Z} = 0
        xz = apply_gate(apply_gate(state, X, [0], 1), Z, [0], 1)
        zx = apply_gate(apply_gate(state, Z, [0], 1), X, [0], 1)
        anticommutator = xz + zx
        self.assert_tensors_close(anticommutator, torch.zeros_like(anticommutator))

        # Y and Z anticommute: {Y,Z} = 0
        yz = apply_gate(apply_gate(state, Y, [0], 1), Z, [0], 1)
        zy = apply_gate(apply_gate(state, Z, [0], 1), Y, [0], 1)
        anticommutator = yz + zy
        self.assert_tensors_close(anticommutator, torch.zeros_like(anticommutator))

class TestCircuits(TestHelper):
    def test_three_CNOTs_simulate_SWAP(self):
        state = zero_state(2)
        state[0] = 0; state[1] = 1  # |01>
        # Apply CNOT(0,1), CNOT(1,0), CNOT(0,1)
        state = apply_gate(state, CNOT, [0, 1], 2)
        state = apply_gate(state, NOTC, [0, 1], 2)
        state = apply_gate(state, CNOT, [0, 1], 2)
        expected = torch.zeros(4, dtype=torch.cfloat)
        expected[2] = 1  # |10>
        self.assert_tensors_close(state, expected)

    def test_SWAP_with_three_CNOTs(self):
        state = zero_state(2)
        state[0] = 0;state[1] = 1  # |01>
        # Apply SWAP
        swapped_state = apply_gate(state, SWAP, [0, 1], 2)
        # Apply three CNOTs
        cnot_state = apply_gate(state, CNOT, [0, 1], 2)
        cnot_state = apply_gate(cnot_state, NOTC, [0, 1], 2)
        cnot_state = apply_gate(cnot_state, CNOT, [0, 1], 2)
        self.assert_tensors_close(swapped_state, cnot_state)

    def test_reversible_circuit(self):
        state = random_state(2)
        # Apply H on 0, CNOT 0->1
        after = apply_gate(state, H, [0], 2)
        after = apply_gate(after, CNOT, [0, 1], 2)
        # Reverse: CNOT 0->1, H on 0
        reversed = apply_gate(after, CNOT, [0, 1], 2)
        reversed = apply_gate(reversed, H, [0], 2)
        self.assert_tensors_close(state, reversed)

    def test_tensor_logic_3_qubit_x1(self):
        # Test applying gates on different qubits
        state = zero_state(3)
        state = apply_gate(state, X, [0], 3) # |100>
        # Apply X first qubit 000 001 010 011 100
        expected = torch.zeros(8, dtype=torch.cfloat)
        expected[4] = 1  # |100>
        self.assert_tensors_close(state, expected)

    def test_tensor_logic_3_qubit_x3(self):
        # Test applying X on qubit 2 (highest index)
        state = zero_state(3)
        # Apply X on qubit 3 (index 0,1,2)
        state = apply_gate(state, X, [2], 3)
        expected = torch.zeros(8, dtype=torch.cfloat)
        expected[1] = 1  # |001>
        self.assert_tensors_close(state, expected)
        
    def test_apply_X2(self):
        # Illustrate applying X gate to qubit 2 in a 3-qubit |000> state
        state = zero_state(3)  # |000>
        new_state = apply_gate(state, X, [1], 3)  # Flips qubit 2: |010>
        expected = torch.zeros(8, dtype=torch.cfloat)
        expected[2] = 1  # |010>
        self.assert_tensors_close(new_state, expected)
     
class TestKraus(TestHelper):
    def test_state_to_density_0(self):
        state = state_to_density(zero_state(1)) # |0><0|
        expected = torch.tensor([[1,0], [0,0]], dtype=torch.cfloat)
        self.assert_tensors_close(state, expected)
        
    def test_state_to_density_plus(self):
        plus = state_to_density(torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)], dtype=torch.cfloat))
        expected = torch.tensor([[1/2,1/2], [1/2,1/2]], dtype=torch.cfloat)
        self.assert_tensors_close(plus, expected)
         
    def test_state_to_density_i(self):
        state = zero_state(1)
        state[0] *= 1j
        state = state_to_density(state)
        expected = torch.tensor([[1,0], [0,0]], dtype=torch.cfloat)
        self.assert_tensors_close(state, expected)
        
    def test_bitflip_kraus(self):
        p_flip = .5
        p_noerr = 1 - p_flip
        ops_probs = [(I2, p_noerr), (X, p_flip)]
        init_state = state_to_density(zero_state(1))
        post_state = kraus_operator(init_state,ops_probs)
        # ρ = |0><0| = [[1,0],[0,0]]
        # XρX = [[0,0], [0,1]]
        # post_ρ = .5 ρ + .5 XρX
        expected = torch.tensor([[.5,0], [0,.5]], dtype=torch.cfloat)
        self.assert_tensors_close(post_state, expected)

    def test_dephasing_kraus(self):
        p = 0.2  # dephasing probability
        # Dephasing Kraus: (1-p) I ρ I + p Z ρ Z
        ops_probs = [(I2, 1 - p), (Z, p)]
        # Use |+> state, which has off-diagonal elements
        plus_state = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)], dtype=torch.cfloat)
        init_density = state_to_density(plus_state)
        post_density = kraus_operator(init_density, ops_probs)
        # Expected: off-diagonals become 0.5 * (1 - 2*p) due to dephasing
        expected = torch.tensor([[0.5, 0.5 * (1 - 2*p)], [0.5 * (1 - 2*p), 0.5]], dtype=torch.cfloat)
        self.assert_tensors_close(post_density, expected)

    def test_depolarizing_kraus(self):
        p = 0.1  # depolarizing parameter
        # Depolarizing Kraus: (1-3p/4) I ρ I + (p/4) X ρ X + (p/4) Y ρ Y + (p/4) Z ρ Z
        ops_probs = [(I2, 1 - 3*p/4), (X, p/4), (Y, p/4), (Z, p/4)]
        # Use |0> state
        init_density = state_to_density(zero_state(1))
        post_density = kraus_operator(init_density, ops_probs)
        # Expected: [[1 - p/2, 0], [0, p/2]]
        expected = torch.tensor([[1 - p/2, 0], [0, p/2]], dtype=torch.cfloat)
        self.assert_tensors_close(post_density, expected)
      
class TestNoiseHelpers(TestHelper):

    # ───────────────────────────────────────────────────────────
    # thermal_relaxation_error_rate
    # ───────────────────────────────────────────────────────────
    def test_thermal_relaxation_zero_idle(self):
        λ1, λ2 = thermal_relaxation_error_rate(T1=30e-6, T2=20e-6, idle_time=0.0)
        self.assertAlmostEqual(λ1, 0.0)
        self.assertAlmostEqual(λ2, 0.0)

    def test_thermal_relaxation_finite_idle(self):
        T1 = 10.0
        T2 = 8.0
        dt = 1.0
        λ1, λ2 = thermal_relaxation_error_rate(T1, T2, dt)

        # Check λ1 matches 1 - exp(-dt/T1)
        self.assertAlmostEqual(λ1, 1.0 - math.exp(-dt / T1))

        # Check that total coherence factor matches e^{-dt/T2}
        # predicted: e^{-dt/(2T1)} (1 - 2 λ2)
        coh_factor = math.exp(-dt / T2)
        coh_factor_pred = math.exp(-dt / (2.0 * T1)) * (1.0 - 2.0 * λ2)
        self.assertAlmostEqual(coh_factor, coh_factor_pred, places=7)

    # ───────────────────────────────────────────────────────────
    # amplitude_damping_kraus
    # ───────────────────────────────────────────────────────────
    def test_amplitude_damping_identity_for_zero_lambda(self):
        ks = amplitude_damping_kraus(0.0)
        self.assertEqual(len(ks), 1)
        self.assert_tensors_close(ks[0], torch.eye(2, dtype=torch.cfloat))

    def test_amplitude_damping_full_relaxation(self):
        # λ1 = 1 ⇒ |1> always decays to |0>
        ks = amplitude_damping_kraus(1.0)
        ρ1 = state_to_density(torch.tensor([0.0, 1.0], dtype=torch.cfloat))  # |1><1|
        out = sum(K @ ρ1 @ torch.adjoint(K) for K in ks)
        expected = torch.tensor([[1.0, 0.0],
                                 [0.0, 0.0]], dtype=torch.cfloat)  # |0><0|
        self.assert_tensors_close(out, expected)

        # completeness: Σ K†K = I
        comp = sum(torch.adjoint(K) @ K for K in ks)
        self.assert_tensors_close(comp, torch.eye(2, dtype=torch.cfloat))

    # ───────────────────────────────────────────────────────────
    # phase_damping_kraus
    # ───────────────────────────────────────────────────────────
    def test_phase_damping_identity_for_zero_lambda(self):
        ks = phase_damping_kraus(0.0)
        self.assertEqual(len(ks), 1)
        self.assert_tensors_close(ks[0], torch.eye(2, dtype=torch.cfloat))

    def test_phase_damping_kills_coherence_for_half(self):
        # λ2 = 0.5 ⇒ off-diagonals multiplied by (1 - 2λ2) = 0
        ks = phase_damping_kraus(0.5)
        plus = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)], dtype=torch.cfloat)
        ρ = state_to_density(plus)
        out = sum(K @ ρ @ torch.adjoint(K) for K in ks)
        expected = torch.tensor([[0.5, 0.0],
                                 [0.0, 0.5]], dtype=torch.cfloat)
        self.assert_tensors_close(out, expected)

        # completeness
        comp = sum(torch.adjoint(K) @ K for K in ks)
        self.assert_tensors_close(comp, torch.eye(2, dtype=torch.cfloat))

class TestKrausHelpers(TestHelper):

    # ───────────────────────────────────────────────────────────
    # make_operations_probs_from_kraus
    # ───────────────────────────────────────────────────────────
    def test_make_operations_probs_from_kraus_two_ops(self):
        K0 = torch.eye(2, dtype=torch.cfloat)
        K1 = X
        ops_probs = make_operations_probs_from_kraus([K0, K1])

        # Should be length 2, with probabilities 1/2 each
        self.assertEqual(len(ops_probs), 2)
        for _, p in ops_probs:
            self.assertAlmostEqual(p, 0.5)

        # Check that the map matches Σ K ρ K†
        ρ = state_to_density(zero_state(1))
        via_ops_probs = kraus_operator(ρ, ops_probs)
        via_direct = K0 @ ρ @ torch.adjoint(K0) + K1 @ ρ @ torch.adjoint(K1)
        self.assert_tensors_close(via_ops_probs, via_direct)

    # ───────────────────────────────────────────────────────────
    # embed_single_qubit_kraus
    # ───────────────────────────────────────────────────────────
    def test_embed_single_qubit_kraus_on_three_qubits(self):
        ks = [X]  # single-qubit X
        # embed on qubit 1 of 3-qubit system
        full_ks = embed_single_qubit_kraus(ks, qubit=1, num_qubits=3)
        self.assertEqual(len(full_ks), 1)
        K_full = full_ks[0]

        # dimension should be 8x8
        self.assertEqual(K_full.shape, (8, 8))

        # Apply to basis state |010> (index 2) → should flip middle qubit → |000>?
        # mapping: qubit 0 is least significant index in your apply_gate; we
        # assume the same order is used consistently here.
        basis = torch.zeros(8, dtype=torch.cfloat)
        basis[2] = 1.0 + 0j  # |010> in little-endian
        out = K_full @ basis

        # X on qubit 1: |010> ↦ |000> (index 0)
        expected = torch.zeros(8, dtype=torch.cfloat)
        expected[0] = 1.0 + 0j
        self.assert_tensors_close(out, expected)

    # ───────────────────────────────────────────────────────────
    # is_only_noise_op / op_time / affected_qubits
    # ───────────────────────────────────────────────────────────
    def test_is_only_noise_op(self):
        self.assertTrue(is_only_noise_op("T1T2_NOISE"))
        self.assertFalse(is_only_noise_op("H"))
        self.assertFalse(is_only_noise_op("CNOT"))

    def test_op_time_lookup(self):
        times = {"H": 1.0, "CNOT": 2.0}
        self.assertAlmostEqual(op_time("H", times), 1.0)
        self.assertAlmostEqual(op_time("CNOT", times), 2.0)
        self.assertAlmostEqual(op_time("Z", times), 0.0)  # default

    # ───────────────────────────────────────────────────────────
    # add_time_based_noise
    # ───────────────────────────────────────────────────────────
    def test_add_time_based_noise_two_qubits(self):
        # Circuit: H on 0, then CNOT on (0,1) with equal durations
        circuit = [("H", [0]), ("CNOT", [0, 1])]
        num_qubits = 2
        T1 = 10.0
        T2 = 8.0
        gate_durations = {"H": 1.0, "CNOT": 1.0}

        noisy = add_time_based_noise(
            circuit=circuit,
            num_qubits=num_qubits,
            T1=T1,
            T2=T2,
            gate_durations=gate_durations,
        )

        # Expected:
        # op0: ("H",[0])
        # op1: noise for q=1, idle_time=1
        # op2: ("CNOT",[0,1])
        self.assertEqual(len(noisy), 3)
        self.assertEqual(noisy[0][0], "H")
        self.assertEqual(noisy[0][1], [0])

        noise_op = noisy[1]
        self.assertEqual(noise_op[0], "T1T2_NOISE")
        self.assertEqual(noise_op[1], [1])
        self.assertAlmostEqual(noise_op[4], 1.0)  # idle_time

        λ1_expected, λ2_expected = thermal_relaxation_error_rate(T1, T2, 1.0)
        self.assertAlmostEqual(noise_op[2], λ1_expected)
        self.assertAlmostEqual(noise_op[3], λ2_expected)

        self.assertEqual(noisy[2][0], "CNOT")
        self.assertEqual(noisy[2][1], [0, 1])

class TestDensityExecutor(TestHelper):
    # ───────────────────────────────────────────────────────────
    # build_full_unitary
    # ───────────────────────────────────────────────────────────
    def test_build_full_unitary_matches_apply_gate(self):
        num_qubits = 2
        gate = H
        targets = [0]

        U_full = build_full_unitary(gate, targets, num_qubits)

        # Check action on a random state matches apply_gate
        state = random_state(num_qubits)
        state_via_full = U_full @ state
        state_via_apply = apply_gate(state, gate, targets, num_qubits)

        self.assert_tensors_close(state_via_full, state_via_apply)

        # Unitarity: U†U = I
        ident = torch.eye(2**num_qubits, dtype=torch.cfloat)
        self.assert_tensors_close(
            torch.adjoint(U_full) @ U_full, ident
        )

    # ───────────────────────────────────────────────────────────
    # apply_named_gate_density
    # ───────────────────────────────────────────────────────────
    def test_apply_named_gate_density_single_qubit(self):
        num_qubits = 1
        ρ0 = state_to_density(zero_state(1))  # |0><0|
        op = ("X", [0])

        ρ1 = apply_named_gate_density(ρ0, op, num_qubits, unitary_cache={})

        expected = state_to_density(torch.tensor([0.0, 1.0], dtype=torch.cfloat))
        self.assert_tensors_close(ρ1, expected)

    # ───────────────────────────────────────────────────────────
    # apply_T1T2_noise_op
    # ───────────────────────────────────────────────────────────
    def test_apply_T1T2_noise_full_relaxation(self):
        num_qubits = 1
        # Start in |1><1|
        ρ1 = state_to_density(torch.tensor([0.0, 1.0], dtype=torch.cfloat))

        # λ1=1 ⇒ full amplitude damping, λ2=0
        noise_op = ("T1T2_NOISE", [0], 1.0, 0.0, 1.0)
        out = apply_T1T2_noise_op(ρ1, noise_op, num_qubits)

        expected = state_to_density(zero_state(1))  # |0><0|
        self.assert_tensors_close(out, expected)

    # ───────────────────────────────────────────────────────────
    # run_noisy_circuit_density
    # ───────────────────────────────────────────────────────────
    def test_run_noisy_circuit_density_no_noise_limit(self):
        # T1, T2 very large ⇒ essentially no noise for small durations
        n = 1
        state = zero_state(n)
        circuit = [("H", [0])]
        gate_durations = {"H": 1e-9}
        T1 = 1e9
        T2 = 1e9

        ρ_final = run_noisy_circuit_density(
            initial_state=state,
            circuit=circuit,
            num_qubits=n,
            T1=T1,
            T2=T2,
            gate_durations=gate_durations,
        )

        plus = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)], dtype=torch.cfloat)
        expected = state_to_density(plus)
        self.assert_tensors_close(ρ_final, expected)

    def test_apply_T1T2_noise_op_trace_preserving(self):
        """
        Previously, combining T1 and T2 Kraus operators incorrectly
        doubled the trace. This test ensures the fixed implementation
        is trace-preserving.
        """
        num_qubits = 1

        # Start from a generic pure state (not an eigenstate of Z)
        state = torch.tensor(
            [1/math.sqrt(3), math.sqrt(2)/math.sqrt(3)],
            dtype=torch.cfloat
        )
        ρ = state_to_density(state)

        # Choose nontrivial λ1 and λ2 so both amplitude and phase damping act
        λ1 = 0.3
        λ2 = 0.4
        idle_time = 1.0
        noise_op = ("T1T2_NOISE", [0], λ1, λ2, idle_time)

        ρ_out = apply_T1T2_noise_op(ρ, noise_op, num_qubits)

        # Check trace is still 1 (within numerical tolerance)
        tr_in = torch.trace(ρ).real.item()
        tr_out = torch.trace(ρ_out).real.item()

        self.assertAlmostEqual(tr_in, 1.0, places=6)
        self.assertAlmostEqual(tr_out, 1.0, places=6)

    def test_run_noisy_circuit_density_trace_preserving(self):
        """
        End-to-end check: the full noisy circuit executor
        should also preserve trace.
        """
        n = 3
        state = zero_state(n)

        circuit = [
            ("H", [0]),
            ("CNOT", [0, 1]),
        ]

        gate_durations = {
            "H": 20e-9,
            "CNOT": 40e-9,
        }

        T1 = 30e-6
        T2 = 20e-6

        ρ_final = run_noisy_circuit_density(
            initial_state=state,
            circuit=circuit,
            num_qubits=n,
            T1=T1,
            T2=T2,
            gate_durations=gate_durations,
        )

        tr_final = torch.trace(ρ_final).real.item()
        self.assertAlmostEqual(tr_final, 1.0, places=6)

    def test_run_noisy_circuit_density_trace_preserving_high_err(self):
        """
        End-to-end check: the full noisy circuit executor
        should also preserve trace.
        """
        n = 3
        state = zero_state(n)

        circuit = [
            ("H", [0]),
            ("CNOT", [0, 1]),
        ]

        gate_durations = {
            "H": 1,
            "CNOT": 1,
        }

        T1 = 10
        T2 = 20

        ρ_final = run_noisy_circuit_density(
            initial_state=state,
            circuit=circuit,
            num_qubits=n,
            T1=T1,
            T2=T2,
            gate_durations=gate_durations,
        )

        tr_final = torch.trace(ρ_final).real.item()
        self.assertAlmostEqual(tr_final, 1.0, places=6)

        
if __name__ == '__main__':
    unittest.main()
