# H7 Metriplectic QML: Physically Grounding AI Decisions with Gemma 4

## The Challenge: The "Black Box" of Algorithmic Governance
As artificial intelligence increasingly integrates into civic operations—such as analyzing municipal parking violations, resource allocation, and urban management—a critical flaw remains: the "Black Box" problem. When an AI issues a penalty or makes a classification that affects citizens (e.g., determining whether a parking violation is classified as "Hostile" or "Clerical Error"), the lack of explainability erodes public trust. 

We built our solution for the **Safety & Trust** track of the **Gemma 4 Impact Challenge**. Our goal is to reimagine algorithmic decision-making not as an opaque statistical output, but as a deterministic, physically measurable, and fully transparent process.

## Our Approach: The Metriplectic Mandate
To bridge the gap between AI intuition and mathematical accountability, we introduced the **H7 Metriplectic Framework**. This framework forces the AI's "thought process" out of the black box and into a simulated physical phase space governed by two forces:
1. **Conservatism (Energy / Symplectic):** The inertia of the data.
2. **Dissipation (Entropy / Metric):** The relaxation of the data towards a stable, understandable conclusion.

Instead of letting an LLM decide the final classification, we use **Gemma 4 (26B)** strictly as an **Information Oracle**. Gemma analyzes the real-world context (the text of the violation) and extracts physical densities ($\rho$, or informational charge) and velocities ($v$, or intentional flow).

These initial conditions are then injected into our Quantum Machine Learning (QML) Lagrangian. The system evolves mathematically until it collapses into one of three Ternary Attractors:
- **Constructive (+1):** Positive or mitigating circumstances.
- **Equilibrium (0):** Neutral administrative records.
- **Destructive (-1):** Malicious, dangerous, or highly disruptive infractions.

## How We Used Gemma 4
We deployed the **Gemma 4 26B** model via Vertex AI. Its unparalleled reasoning capabilities allow it to process unstructured civic data (e.g., "Vehicle double-parked on a narrow street blocking emergency access") and map it to continuous physical parameters. 

### Why Gemma 4?

Gemma 4 provided the perfect balance of semantic nuance and strict output formatting required to act as our Metriplex Oracle. We relied on its instruction-tuned variant (`gemma-4-26b-a4b-it`) strictly deployed via Vertex AI to natively parse varied, noisy parking ticket descriptions into precise JSON vectors ($\rho$ and $v$) without hallucinating extra dimensions.

### Global Resilience: ONNX Edge Offline Fallback

To directly address the challenge's goal of building for contexts like **"an offline, edge-based disaster response"**, our architecture refuses to rely entirely on an internet connection. We implemented a **True Edge Fallback via ONNX Runtime**.
If the system detects a network drop to the Google Cloud Vertex API, the `gemma_oracle.py` automatically initializes **Gemma 2B/4B** models locally. By utilizing the `optimum.onnxruntime` library, the model weights are loaded dynamically into CPU/constrained memory on the edge device itself (e.g., a local municipal tablet or Raspberry Pi). 
This guarantees that life-critical algorithmic governance remains functional and mathematically accountable even in completely unplugged environments.

## Technical Architecture
1. **Data Ingestion:** Processing civic datasets of parking violations.
2. **Oracle Extraction (Gemma 4):** Translating text descriptions into physical states ($\rho \in [0.1, 1.0]$, $v \in [-1.0, 1.0]$).
3. **Phase Space Modulation ($O_n$):** The space is modulated by the Golden Operator ($O_n = \cos(\pi n) \cdot \cos(\pi \phi n)$) ensuring stability.
4. **H7 QML Classifier:** Evaluates the Metriplectic Lagrangian over a time step $t$, visualizing the convergence of energy and entropy down to the final predicted ternary state.
5. **Dashboard:** A Streamlit interface that proves the "Wow Factor," allowing auditors to dynamically visualize the decision path of the AI before it officially classifies the ticket.

## Why Technical Choices Matter
By adopting a Metriplectic dynamic for classification rather than a traditional Cross-Entropy loss neural network, we achieved **transparent execution**. If a citizen challenges why their parking ticket was classified as "Destructive" (-1), an auditor does not refer to hidden weights in a GPU; they point to the explicitly calculated trajectory of the informational physics proving that the entropy of the semantic act pulled the system toward the negative attractor.

This hybrid architecture—pairing the abstract reasoning of frontier intelligence (Gemma 4) with the strict, unbreakable accountability of physics-inspired logic—is the future of algorithmic governance. It paves the way for AI systems that don't just ask us to trust them, but *prove* why we should.
