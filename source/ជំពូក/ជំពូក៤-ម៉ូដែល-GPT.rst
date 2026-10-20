.. meta::
   :description: ជំពូក ៤ ការអនុវត្តម៉ូដែល GPT ពីមូលដ្ឋាន — ស្ថាបត្យកម្មម៉ូដែល, KV cache, GQA, MLA, SWA, MoE។ Chapter 4 Implementing a GPT Model from Scratch — architecture, KV cache, GQA, MLA, SWA, MoE.
   :keywords: ជំពូក ៤, GPT, from scratch, gpt.py, KV cache, GQA, MLA, SWA, MoE, DeltaNet, DSA, FLOPs

ជំពូក ៤៖ ការអនុវត្តម៉ូដែល GPT ពីមូលដ្ឋាន
====================================================

.. note::

   ប្រភព៖ ថត `ch04 <https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04>`_
   និងតារាងមាតិកាក្នុង `README.md <https://github.com/rasbt/LLMs-from-scratch/blob/main/README.md>`_ របស់ upstream។

ជំពូកនេះជាជំហានសំខាន់ — ការផ្គុំបំណែកទាំងអស់ចេញជាម៉ូដែល GPT ពេញលេញ
ដែលអនុវត្តពីមូលដ្ឋានដោយប្រើ PyTorch។ ក្រៅពីកូដមេ ជំពូកនេះមានសម្ភារៈបន្ថែម
ច្រើនជាងគេអំពីជម្រើសស្ថាបត្យកម្ម Attention និងការវិភាគប្រសិទ្ធភាព។

កូដមេ
-------

កូដមេស្ថិតក្នុងថត
`ch04/01_main-chapter-code <https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/01_main-chapter-code>`_៖

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - ឯកសារ
     - តួនាទី
   * - `ch04.ipynb <https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/ch04.ipynb>`_
     - notebook មេរបស់ជំពូក
   * - `gpt.py <https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/gpt.py>`_
     - សង្ខេប (summary) ម៉ូដែល GPT ជាឯកសារ Python
   * - `exercise-solutions.ipynb <https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/exercise-solutions.ipynb>`_
     - ចម្លើយលំហាត់របស់ជំពូក

Notebooks ខ្មែរសម្រាប់ជំពូកនេះ
-----------------------------

ឃ្លាំងឯកសារនេះមានកំណែ notebook ជាភាសាខ្មែរ (ថត ``ណូតប៊ុក-ខ្មែរ/``)៖

- `ជំពូក៤-ម៉ូដែល-GPT.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៤-ម៉ូដែល-GPT.ipynb>`_ — notebook មេ ព្រមទាំងការពន្យល់ និង comments ជាខ្មែរ
- `ជំពូក៤-ចម្លើយលំហាត់.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៤-ចម្លើយលំហាត់.ipynb>`_ — ចម្លើយលំហាត់របស់ជំពូក

សូមមើលបញ្ជីពេញលេញ និងរបៀបដំណើរការក្នុងទំព័រ :doc:`../យោង/ណូតប៊ុក-ខ្មែរ`។

សម្ភារៈបន្ថែម និងជម្រើស Attention
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - ថត
     - ខ្លឹមសារ
   * - ``02_performance-analysis``
     - ការវិភាគ FLOPs (``flops-analysis.ipynb``)
   * - ``03_kv-cache``
     - KV Cache
   * - ``04_gqa``
     - Grouped-Query Attention (GQA)
   * - ``05_mla``
     - Multi-Head Latent Attention (MLA)
   * - ``06_swa``
     - Sliding Window Attention (SWA)
   * - ``07_moe``
     - Mixture-of-Experts (MoE)
   * - ``08_deltanet``
     - Gated DeltaNet
   * - ``09_dsa``
     - DeepSeek Sparse Attention (DSA)
   * - ``10_kv-sharing``
     - Cross-Layer KV Sharing

.. note::

   បញ្ជីខាងលើឆ្លុះបញ្ចាំងពីរចនាសម្ព័ន្ធថតពិត និងបញ្ជីសម្ភារៈបន្ថែមក្នុង ``README.md``
   របស់ upstream។ លម្អិតបច្ចេកទេសនៃវិធីសាស្ត្រនីមួយៗ សូមអានក្នុង notebooks
   ផ្លូវការនៅឃ្លាំង upstream។

លំហាត់
--------

ចម្លើយលំហាត់មានក្នុង ``exercise-solutions.ipynb`` ខាងលើ និងក្នុងសង្ខេបនៃភាគបន្ថែម C។

.. tip::

   បន្ទាប់៖ បណ្តុះបណ្តាលម៉ូដែលឱ្យដំណើរការពិតប្រាកដក្នុង
   :doc:`ជំពូក៥-pretraining`។
