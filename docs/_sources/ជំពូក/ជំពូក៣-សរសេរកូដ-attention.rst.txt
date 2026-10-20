.. meta::
   :description: ជំពូក ៣ ការសរសេរកូដ Attention Mechanisms — self-attention, multi-head attention និងការអនុវត្តប្រកបដោយប្រសិទ្ធភាព។ Chapter 3 Coding Attention Mechanisms — self-attention, multi-head attention and efficient implementations.
   :keywords: ជំពូក ៣, Coding Attention Mechanisms, attention, multi-head attention, buffers, PyTorch

ជំពូក ៣៖ ការសរសេរកូដ Attention Mechanisms
=================================================

.. note::

   ប្រភព៖ ថត `ch03 <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch03>`_
   និងតារាងមាតិកាក្នុង `README.md <https://github.com/rasbt/LLMs-from-scratch/blob/main/README.md>`_ របស់ upstream។

ជំពូកនេះណែនាំការសរសេរកូដយន្តការ Attention — សមាសភាគស្នូលដែលធ្វើឱ្យ LLM
អាចយកចិត្តទុកដាក់លើផ្នែកពាក់ព័ន្ធនៃទិន្នន័យបញ្ចូល រួមទាំងការអនុវត្ត Multi-Head Attention។

មេរៀនសង្ខេប (សិក្សាងាយ)
--------------------------

ការពន្យល់ខាងក្រោមជាអត្ថបទដើមរបស់សហគមន៍ ដើម្បីជួយឱ្យយល់គំនិតសំខាន់ៗមុនពេលអានកូដ៖

- **គំនិត attention** — ជំនួសឱ្យការមើលគ្រប់ទីតាំងស្មើៗគ្ា ម៉ូដែល "ថ្លឹង" ថា token ណាខ្លះសំខាន់សម្រាប់បង្កើតការតំណាងរបស់ token នីមួយៗ។
- **Dot-product score** — រង្វាស់ភាពស្រដៀងគ្នារវាង query និង key។ កាន់តែខ្ពស់ កាន់តែត្រូវយកចិត្តទុកដាក់។
- **Softmax** — បំលែង scores ទៅជាទម្ងន់ដែលបូកសរុបស្មើ ១។
- **Causal mask** — ទប់មិនឱយ token មើលឃើញ tokens នៅខាងមុខវា ដើម្បីឱ្យការបង្កើតអត្ថបទត្រឹមត្រូវ។
- **Q/K/V projections** — ម៉ាទ្រីសទម្ងន់ដែលអាចរៀនបាន សម្រាប់បង្កើត query, key, value របស់ token នីមួយៗ។
- **Multi-head attention** — ដំណើរការ attention ច្រើន "ក្បាល" ជាមួយគ្នា ដើម្បីមើលទិន្នន័យពីទស្សនវិស័យច្រើន។

ដំណើរការកូដក្នុងជំពូកនេះ៖

1. គណនា attention scores ដៃ និង context vector ដំបូង។
2. ណែនាំម៉ាទ្រីស Q/K/V ដែលរៀនបាន។
3. បង្កប់ទាំងអស់ចូលក្លាស SelfAttention_v2។
4. បន្ថែម causal mask និង dropout (CausalAttention)។
5. ប្រៀបធៀបការអនុវត្ត multi-head attention ពីរបែប។

កូដមេ
-------

កូដមេស្ថិតក្នុងថត
`ch03/01_main-chapter-code <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch03/01_main-chapter-code>`_៖

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - ឯកសារ
     - តួនាទី
   * - `ch03.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៣-យន្តការ-attention.ipynb>`_
     - notebook មេរបស់ជំពូក
   * - `multihead-attention.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៣-សង្ខេប-multihead-attention.ipynb>`_
     - សង្ខេប (summary) នៃ multi-head attention
   * - `exercise-solutions.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៣-ចម្លើយលំហាត់.ipynb>`_
     - ចម្លើយលំហាត់របស់ជំពូក

Notebooks ខ្មែរសម្រាប់ជំពូកនេះ
-----------------------------

ឃ្លាំងឯកសារនេះមានកំណែ notebook ជាភាសាខ្មែរ (ថត ``ណូតប៊ុក-ខ្មែរ/``)៖

- `ជំពូក៣-យន្តការ-attention.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៣-យន្តការ-attention.ipynb>`_ — notebook មេ ព្រមទាំងការពន្យល់ និង comments ជាខ្មែរ
- `ជំពូក៣-សង្ខេប-multihead-attention.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៣-សង្ខេប-multihead-attention.ipynb>`_ — សង្ខេប multi-head attention + data loader
- `ជំពូក៣-ចម្លើយលំហាត់.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៣-ចម្លើយលំហាត់.ipynb>`_ — ចម្លើយលំហាត់របស់ជំពូក

សូមមើលបញ្ជីពេញលេញ និងរបៀបដំណើរការក្នុងទំព័រ :doc:`../យោង/ណូតប៊ុក-ខ្មែរ`។

រចនាសម្ព័ន្ធថត
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ថត
     - ខ្លឹមសារ
   * - ``01_main-chapter-code``
     - កូដមេរបស់ជំពូក
   * - ``02_bonus_efficient-multihead-attention``
     - ប្រៀបធៀបការអនុវត្ត Multi-Head Attention ដែលមានប្រសិទ្ធភាព (``mha-implementations.ipynb``)
   * - ``03_understanding-buffers``
     - ស្វែងយល់អំពី PyTorch Buffers (``understanding-buffers.ipynb``)

លំហាត់
--------

ចម្លើយលំហាត់មានក្នុង ``exercise-solutions.ipynb`` ខាងលើ និងក្នុងសង្ខេបនៃភាគបន្ថែម C។

.. tip::

   បន្ទាប់៖ យកចំណេះដឹង Attention ទៅសាងសង់ម៉ូដែល GPT ពេញលេញក្នុង
   :doc:`ជំពូក៤-ម៉ូដែល-GPT`។
