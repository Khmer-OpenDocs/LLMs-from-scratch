.. meta::
   :description: ជំពូក ៤ ការអនុវត្តម៉ូដែល GPT ពីមូលដ្ឋាន — ស្ថាបត្យកម្មម៉ូដែល, KV cache, GQA, MLA, SWA, MoE។ Chapter 4 Implementing a GPT Model from Scratch — architecture, KV cache, GQA, MLA, SWA, MoE.
   :keywords: ជំពូក ៤, GPT, from scratch, gpt.py, KV cache, GQA, MLA, SWA, MoE, DeltaNet, DSA, FLOPs

ជំពូក ៤៖ ការអនុវត្តម៉ូដែល GPT ពីមូលដ្ឋាន
====================================================

.. note::

   ប្រភព៖ ថត `ch04 <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch04>`_
   និងតារាងមាតិកាក្នុង `README.md <https://github.com/rasbt/LLMs-from-scratch/blob/main/README.md>`_ របស់ upstream។

ជំពូកនេះជាជំហានសំខាន់ — ការផ្គុំបំណែកទាំងអស់ចេញជាម៉ូដែល GPT ពេញលេញ
ដែលអនុវត្តពីមូលដ្ឋានដោយប្រើ PyTorch។ ក្រៅពីកូដមេ ជំពូកនេះមានសម្ភារៈបន្ថែម
ច្រើនជាងគេអំពីជម្រើសស្ថាបត្យកម្ម Attention និងការវិភាគប្រសិទ្ធភាព។

មេរៀនសង្ខេប (សិក្សាងាយ)
--------------------------

ការពន្យល់ខាងក្រោមជាអត្ថបទដើមរបស់សហគមន៍ ដើម្បីជួយឱ្យយល់គំនិតសំខាន់ៗមុនពេលអានកូដ៖

- **Transformer block** — ឯកតាដែលត្រួតគ្នា៖ LayerNorm → Attention → LayerNorm → Feed-Forward ព្រំទាំង residual connections។
- **LayerNorm** — ធ្វើឱ្យមធ្យម និងវ៉ារ្យង់នៃវ៉ិចទ័រនីមួយៗមានស្ថិរភាព ជួយឱ្យការបណ្តុះបណ្តាលងាយស្រួល។
- **GELU និង FFN** — បណ្តាញពីរស្រទាប់ដែលមាន activation រលូន បន្ថែមភាពមិនលីនេអ៊ែរទៅក្នុងម៉ូដែល។
- **Residual connection** — ផ្លូវកាត់ដែលឱ្យ gradient ហូរបានល្អ និងអនុញ្ញាតឱ្យដាក់ស្រទាប់ច្រើន។
- **Config និងចំនួន parameters** — តម្លៃដូចជា vocab size, emb_dim, n_layers, n_heads, context length កំណត់ទំហំ និងសមត្ថភាពម៉ូដែល។
- **វដ្តបង្កើតអត្ថបទ** — យក logits នៃទីតាំងចុងក្រោយ → ជ្រើស token → បន្ថែមចូល context → ធ្វើម្តងទៀត។

ដំណើរការកូដក្នុងជំពូកនេះ៖

1. កំណត់ config (GPT_CONFIG_124M) និងត្រៀមបញ្ចូលឧទាហរណ៍។
2. សាងសង់ LayerNorm និង GELU ដោយដៃ បន្ទាប់មកជាច្លាស។
3. សាងសង់ FeedForward និងឧទាហរណ៍ residual connections។
4. ផ្គុំ TransformerBlock និង GPTModel ពេញលេញ។
5. រាប់ parameters និងប៉ាន់ស្មានអង្គចងចាំ។
6. សរសេរ generate_text_simple សម្រាប់ inference។

កូដមេ
-------

កូដមេស្ថិតក្នុងថត
`ch04/01_main-chapter-code <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch04/01_main-chapter-code>`_៖

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - ឯកសារ
     - តួនាទី
   * - `ch04.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៤-ម៉ូដែល-GPT.ipynb>`_
     - notebook មេរបស់ជំពូក
   * - `gpt.py <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/កូដ/ch04/01_main-chapter-code/gpt.py>`_
     - សង្ខេប (summary) ម៉ូដែល GPT ជាឯកសារ Python
   * - `exercise-solutions.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៤-ចម្លើយលំហាត់.ipynb>`_
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
