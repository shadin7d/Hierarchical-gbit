"""
Phase S: Physics Relaxation Solver
Tests the core double-well relaxation dynamics.
"""

import sys
sys.path.insert(0, '.')

from gbit import HierarchicalGBit, set_seed

def run():
    """Run Phase S relaxation benchmark."""
    print("  Running Phase S — Physics Relaxation Solver...")
    print()
    
    set_seed(42)
    
    results = {
        "name": "Phase S — Relaxation Dynamics",
        "trials": 5,
        "cases": []
    }
    
    # Test relaxation from different initial states
    test_cases = [
        ("Positive well", 1.5),
        ("Negative well", -1.5),
        ("Unstable (origin)", 0.1),
    ]
    
    for label, q_init in test_cases:
        print(f"    {label:20s} (q_init={q_init:+.1f}):")
        
        energies = []
        for trial in range(5):
            g = HierarchicalGBit()
            g._inner.q[0] = q_init
            g._inner.v[0] = 0.0
            
            # Relax without external force
            for step in range(2000):
                g.step(tilt=0.0)
            
            # Compute energy at end
            q = g._inner.q[0]
            d = g._inner.d[0]
            E = d * (q**2 - 1)**2
            energies.append(E)
        
        mean_E = sum(energies) / len(energies)
        min_E = min(energies)
        max_E = max(energies)
        
        print(f"      Final Energy: {mean_E:.4f} ± [{min_E:.4f}, {max_E:.4f}]")
        
        results["cases"].append({
            "label": label,
            "mean_energy": mean_E,
            "min_energy": min_E,
            "max_energy": max_E,
        })
    
    print()
    return results
