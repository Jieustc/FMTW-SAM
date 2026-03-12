# FMTW-SAM: Cross-Domain Semi-Supervised Segmentation with SAM

FMTW-SAM is a deep learning framework for cross-domain semi-supervised segmentation of Type-B aortic dissection in computed tomography angiography (CTA) images. It extends the Segment Anything Model (SAM) with foreground-mixing fine-tuning and temporally weighted SAM feature fusion to improve true and false lumen segmentation under limited annotations and domain shift.

The framework is designed for challenging cross-domain scenarios, where CTA data from different medical centers may vary in imaging protocols and patient populations. By combining bidirectional foreground mixing with a dual-student Mean Teacher framework, FMTW-SAM aims to improve segmentation performance when only a small amount of labeled data is available.

This repository currently contains part of the core training code. Additional details, complete training scripts, and related resources will be released in a future update.


# Citation
Citation information will be added after publication.


# Acknowledgement

This code is mainly based on [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [SAM](https://github.com/facebookresearch/segment-anything).

Some of the other code is from [BCP](https://github.com/DeepMed-Lab-ECNU/BCP),  [ABD](https://github.com/Star-chy/ABD).

**Note:** The rest of the details will be released soon. Currently, only part of the core code is provided.
