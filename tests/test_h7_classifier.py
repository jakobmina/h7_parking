import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from h7_qml_classifier import H7TernaryClassifier

def test_rule1_reversibility():
    """
    Regla 1: Ley de la Reversibilidad.
    Si rho -> 1 y v -> constante, el término disipativo debería ser casi cero (fluido ideal),
    y la dinámica debería ser netamente conservativa (oscilante, d_metr aprox 0).
    """
    clf = H7TernaryClassifier(epsilon=0.25)
    psi_test = 0.5
    rho = 1.0 # Densidad máxima
    v = 0.5
    
    d_symp, d_metr = clf.compute_lagrangian(psi_test, rho, v)
    
    # d_symp es la inercia. d_metr es la fricción.
    # Con rho=1.0, el atractor desaparece (1-rho)=0. Queda sólo la modulación Áurea de muy baja fricción.
    assert abs(d_metr) < 0.1, f"Fallo Regla 1: d_metr alto {d_metr} en límite conservativo"
    assert d_symp != 0, f"Fallo Regla 1: d_symp {d_symp} es 0, no hay conservación"

def test_rule2_homogeneidad():
    """
    Regla 2: Homogeneidad Dimensional (Validación de Constantes).
    Verificar que phi es exactamente la Razón Áurea.
    """
    clf = H7TernaryClassifier()
    assert abs(clf.phi - 1.6180339887) < 1e-6, "Fallo Regla 2: Phi (Razón Áurea) incorrecta."

def test_rule3_limite_asintotico():
    """
    Regla 3: Prueba de los Límites Asintóticos.
    Si v -> 0 (Sin intencionalidad o flujo de entrada), y rho -> 0 (Vacío total),
    el sistema debería colapsar asintóticamente a su atractor más cercano 
    puramente por L_metr. (El espacio mismo relaja la entropía - Hawking/Casimir).
    """
    clf = H7TernaryClassifier()
    rho = 0.0
    v = 0.0
    psi_inicial = 0.2
    
    # Ejecutamos ciclo puramente disipativo manualmente
    psi = psi_inicial
    for _ in range(100):
        _, d_metr = clf.compute_lagrangian(psi, rho, v)
        psi += d_metr * 0.1
    
    # Debería relajarse a 0 (El atractor más cercano para 0.2)
    assert abs(psi) < 1e-3, f"Fallo Regla 3: El sistema no colapsó a su atractor, psi={psi}"
