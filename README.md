# Bridging Risk-Sensitive and Multi-Objective Reinforcement Learning: A Comparative Analysis of Dead-end Discovery and Set-Valued Policies in Clinical Decision-Making

## Background & Objective
Machine learning has successfully framed many sequential decision-making problems as either supervised prediction tasks or as the search for optimal policies via reinforcement learning. 
In clinical decision-making—particularly when learning treatment strategies from electronic health records (EHRs)—data constraints and offline settings can pose significant challenges. 
Traditional methods may fail because they assume fully optimal behavior or rely on exploring alternatives that might not be present. 
In safety-critical domains like healthcare, even if optimality is unattainable, the negative outcomes in historical data can be leveraged to identify behaviors to avoid, thereby guarding against overoptimistic decisions.

In recent years, two innovative RL-based approaches have emerged to guide treatment recommendations:
Dead-end Discovery (DeD): A risk-sensitive method that identifies irreversible, high-risk states (e.g., patient deterioration leading to mortality) and eliminates actions likely to steer patients toward these outcomes.
Set-Valued Policies (SVPs): An approach that offers clinicians a set of multiple near-optimal treatment options instead of a single recommendation, maintaining a proportional margin of optimality and allowing for flexible, clinician-in-the-loop decision-making.

Both methods aim to move beyond an authoritative, singular treatment recommendation paradigm. DeD avoids undesirable treatments, while SVPs highlight multiple desirable actions. However, each comes with its limitations: 
DeD can be overly conservative—potentially discarding beneficial treatments due to biases in historical records—whereas SVPs might be over-optimistic, lacking explicit risk-assessment and adaptive margin adjustment in high-risk states.

In this repository, we provide the code for our research project that seeks to design a unified, risk-aware, multi-objective framework integrating the complementary strengths of DeD and SVPs while mitigating their respective limitations.
By combining the risk-avoidance strategy of DeD with the flexibility of SVPs, our framework is designed to provide clinicians with robust, context-sensitive decision support, ultimately aiming to improve patient outcomes in safety-critical settings.
