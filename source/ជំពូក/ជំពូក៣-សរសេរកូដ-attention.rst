.. meta::
   :description: ជំពូក ៣ ការសរសេរកូដ Attention Mechanisms — self-attention, multi-head attention និងការអនុវត្តប្រកបដោយប្រសិទ្ធភាព។ Chapter 3 Coding Attention Mechanisms — self-attention, multi-head attention and efficient implementations.
   :keywords: ជំពូក ៣, Coding Attention Mechanisms, attention, multi-head attention, buffers, PyTorch

ជំពូក ៣៖ ការសរសេរកូដ Attention Mechanisms
=================================================

.. note::

   ប្រភព៖ ថត `ch03 <https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03>`_
   និងតារាងមាតិកាក្នុង `README.md <https://github.com/rasbt/LLMs-from-scratch/blob/main/README.md>`_ របស់ upstream។

ជំពូកនេះណែនាំការសរសេរកូដយន្តការ Attention — សមាសភាគស្នូលដែលធ្វើឱ្យ LLM
អាចយកចិត្តទុកដាក់លើផ្នែកពាក់ព័ន្ធនៃទិន្នន័យបញ្ចូល រួមទាំងការអនុវត្ត Multi-Head Attention។

កូដមេ
-------

កូដមេស្ថិតក្នុងថត
`ch03/01_main-chapter-code <https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03/01_main-chapter-code>`_៖

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - ឯកសារ
     - តួនាទី
   * - `ch03.ipynb <https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/ch03.ipynb>`_
     - notebook មេរបស់ជំពូក
   * - `multihead-attention.ipynb <https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/multihead-attention.ipynb>`_
     - សង្ខេប (summary) នៃ multi-head attention
   * - `exercise-solutions.ipynb <https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/exercise-solutions.ipynb>`_
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
