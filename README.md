# FMTW-SAM: AI-powered Cross-Domain Semi-Supervised Segmentation Module

FMTW-SAM is an advanced deep learning framework designed to enhance the Segment Anything Model (SAM) for cross-domain semi-supervised segmentation of Type-B aortic dissection in computed tomography angiography (CTA) images. By integrating bidirectional foreground mixing and temporally weighted SAM feature fusion, this module delivers robust and reliable true and false lumen segmentation, providing essential support for precise diagnosis and effective surgical planning in clinical practice.

Built with flexibility and performance in mind, FMTW-SAM utilizes innovative techniques such as foreground-mixing fine-tuning and a dual-student Mean Teacher architecture to optimize segmentation under limited annotations and significant domain shifts. It is tailored to work efficiently even in complex cross-domain scenarios, where data from different medical centers exhibit distributional discrepancies in imaging protocols and patient characteristics.

FMTW-SAM aims to reduce the workload of healthcare professionals by enabling state-of-the-art performance with only a small amount of labeled data, making more efficient and accurate clinical decisions possible for Type-B aortic dissection management.

This framework is designed to be seamlessly integrated with SAM, offering an easy-to-use solution for delivering high-quality semi-supervised segmentation across varying CTA domains, making it a versatile tool for a wide range of cardiovascular imaging applications.

# Acknowledgement

This code is mainly based on [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [SAM](https://github.com/facebookresearch/segment-anything).

Some of the other code is from [BCP](https://github.com/DeepMed-Lab-ECNU/BCP),  [ABD](https://github.com/Star-chy/ABD).

**Note:** The rest of the details will be released soon. Currently, only part of the core code is provided.
