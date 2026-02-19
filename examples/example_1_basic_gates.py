"""Example 1: Basic Gate Operations"""
import sys
sys.path.insert(0, '.')
from gbit import HierarchicalGBit

def example_basic_gates():
    print("=" * 60)
    print("EXAMPLE 1: Basic Logic Gates")
    print("=" * 60)
    gates = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR"]
    for gate in gates:
        print(f"{gate} Gate Truth Table:")
        for A in [0, 1]:
            for B in [0, 1]:
                g = HierarchicalGBit()
                result, _, _ = g.compute_gate(A, B, gate)
                print(f"  {A} {gate} {B} = {result}")
        print()

if __name__ == "__main__":
    example_basic_gates()
