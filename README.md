This code is the official implementation of `Amortized Pormpt: Lightweight Fine-Tuning CLIP in Domain Generalizaiton`

<!-- This code is based on the official implementation of `Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization (NIPS2021)`.  -->

<!-- This codebase is mainly based on [DomainBed](https://github.com/facebookresearch/DomainBed), with following modifications: -->

## -- Concet --
![concept](https://user-images.githubusercontent.com/49514261/144587743-9fc28c90-c6d6-4d67-9e3b-02d412c693d0.png)
The conceptual difference between (a) Standard DG methods that using ResNet18 or ResNet50 as a backbone and (b) Foundation
Model such as CLIP. The most of standard DG methods (a) explicit/implicit conduct domain alignment to learn domain-invariant repre-
sentation or add samples/regularization to avoid overfitting on source domains. In this work, we propose (b) to utilize Foundation Model
that includes more effective representation for adaption to a target domain

## -- Architecture --
![architecture](https://user-images.githubusercontent.com/49514261/144587733-010b67b2-b4b3-41e3-876b-1587ef8ba9b7.png)

Architecture for (a) Empirical Risk Minimization (ERM) fine-tuning from prior works, (b) naive CLIP without fine-tuning, and
(c) CLIP + AP. Gray boxes are fixed networks during learning, while blue boxes are the learned components. Instead of intervening
through directly on back-bone vision network representations as in (a) ERM, our (c) CLIP + AP intervenes through prompt generation in
language representation, passed through the backbone language network

## -- results --
![image](https://user-images.githubusercontent.com/49514261/144588234-8764e615-d4ec-4aa4-841c-04f2f1c599fc.png)

Detailed results on VLCS, PACS, OfficeHome, TerraIncognita. The performance of (CLIP + AP) - (CLIP) shows the consistent
improvement over CLIP all of the datasets. We highlight the most improved domain in each dataset.
