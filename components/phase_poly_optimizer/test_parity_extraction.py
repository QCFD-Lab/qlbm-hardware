#!/usr/bin/env python3
"""
Test the parity table extraction from the PhasePolyOptimizer class
using the circuit from Figure 1 of Vandaele2022 paper.

The circuit has phase polynomial:
p(x1,x2,x3,x4) = theta1(x1 XOR x2) + theta2(x1 XOR x2 XOR x3) + theta3(x1 XOR x3 XOR x4)

Expected parity table P:
[[1, 1, 1],
 [1, 1, 0],
 [0, 1, 1],
 [0, 0, 1]]
"""

import numpy as np
from qiskit import QuantumCircuit
from components.phase_poly_optimizer.phase_poly_optimize import PhasePolyOptimizer

def create_test_circuit():
    # Qubit ordering: q0=x1, q1=x2, q2=x3, q3=x4
    circ = QuantumCircuit(4)

    circ.cx(1, 0)

    circ.rz(1.0, 0)  # Using theta1 = 1.0

    circ.cx(2, 0)

    circ.rz(2.0, 0)  # Using theta2 = 2.0

    circ.cx(0, 1)

    circ.cx(3, 1)

    circ.rz(3.0, 1)  # Using theta3 = 3.0

    return circ



def test_parity_extraction():
    """Test the parity table extraction"""

    circ = create_test_circuit()
    print(f"\nCircuit (Qiskit representation):")
    print(circ)

    optimizer = PhasePolyOptimizer(circ)
    blocks = optimizer.find_blocks()
    print(f"\nFound {len(blocks)} phase polynomial blocks")

    if blocks:
        start, end, qubits, gates = blocks[0]
        print(f"Block: start={start}, end={end}, qubits={qubits}")
        print(f"Gates in block: {len(gates)}")

        # Show the gates
        for i, (gate, qargs) in enumerate(gates):
            qubit_indices = [circ.find_bit(q)[0] for q in qargs]
            if gate.name == 'cx':
                print(f"  {i}: CNOT(q{qubit_indices[0]}, q{qubit_indices[1]})")
            elif gate.name == 'rz':
                print(f"  {i}: RZ(q{qubit_indices[0]}, {float(gate.params[0])})")

        # Extract the block and parity table
        block_circuit = optimizer.extract_block(start, end)
        print(f"\nBlock circuit:")
        print(block_circuit)

        P, angles = optimizer._extract_parity_table(block_circuit)
        print(f"\nExtracted parity table P:")
        print(P)
        print(f"Extracted angles: {angles}")

        expected_P = np.array([
            [1, 1, 1],  # x1 appears in all 3 parities
            [1, 1, 0],  # x2 appears in parities 1 and 2
            [0, 1, 1],  # x3 appears in parities 2 and 3
            [0, 0, 1]   # x4 appears only in parity 3
        ])

        expected_angles = [1.0, 2.0, 3.0]  # θ₁, θ₂, θ₃

        print(f"\nExpected parity table P:")
        print(expected_P)
        print(f"Expected angles: {expected_angles}")

        # Check if they match
        if np.array_equal(P, expected_P):
            print("✓ Parity table matches expected!")
        else:
            print("✗ Parity table does not match expected")
            print(f"Difference:")
            print(P - expected_P)

        if np.allclose(angles, expected_angles):
            print("✓ Angles match expected!")
        else:
            print("✗ Angles do not match expected")
            print(f"Angle differences: {np.array(angles) - np.array(expected_angles)}")

        # Test the optimization
        print(f"\n" + "="*60)
        print("Testing circuit optimization...")
        optimized_circ = optimizer.optimize_block(block_circuit)
        print(f"Original circuit depth: {circ.depth()}")
        print(f"Optimized circuit depth: {optimized_circ.depth()}")
        print(f"Original CNOT count: {sum(1 for gate, _, _ in circ.data if gate.name == 'cx')}")
        print(f"Optimized CNOT count: {sum(1 for gate, _, _ in optimized_circ.data if gate.name == 'cx')}")

        print(f"\nOptimized circuit:")
        print(optimized_circ)

    else:
        print("No phase polynomial blocks found!")

if __name__ == "__main__":
    test_parity_extraction()