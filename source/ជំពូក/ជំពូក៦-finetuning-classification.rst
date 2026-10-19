.. meta::
   :description: ជំពូក ៦ Finetuning សម្រាប់ចំណាត់ថ្នាក់អត្ថបទ — spam classifier, dataset IMDb និង UI។ Chapter 6 Finetuning for Text Classification — spam classifier, IMDb dataset and UI.
   :keywords: ជំពូក ៦, finetuning, text classification, gpt_class_finetune.py, spam classifier, IMDb, pandas, matplotlib

ជំពូក ៦៖ Finetuning សម្រាប់ចំណាត់ថ្នាក់អត្ថបទ
======================================================

.. note::

   ប្រភព៖ ថត `ch06 <https://github.com/rasbt/LLMs-from-scratch/tree/main/ch06>`_
   និងតារាងមាតិកាក្នុង `README.md <https://github.com/rasbt/LLMs-from-scratch/blob/main/README.md>`_ របស់ upstream។

ជំពូកនេះបង្ហាញការកែសម្រួល (finetune) ម៉ូដែល GPT ដែលបាន pretrain ហើយ
ឱ្យក្លាយជាម៉ូដែលចំណាត់ថ្នាក់អត្ថបទ (text classification)។ បណ្ណាល័យដែល
``requirements.txt`` កំណត់សម្រាប់ជំពូកនេះ៖ ``matplotlib`` · ``tensorflow`` · ``pandas``។

កូដមេ
-------

កូដមេស្ថិតក្នុងថត
`ch06/01_main-chapter-code <https://github.com/rasbt/LLMs-from-scratch/tree/main/ch06/01_main-chapter-code>`_៖

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - ឯកសារ
     - តួនាទី
   * - `ch06.ipynb <https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/ch06.ipynb>`_
     - notebook មេរបស់ជំពូក
   * - `gpt_class_finetune.py <https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/gpt_class_finetune.py>`_
     - ស្គ្រីប finetuning សម្រាប់ចំណាត់ថ្នាក់
   * - `exercise-solutions.ipynb <https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/exercise-solutions.ipynb>`_
     - ចម្លើយលំហាត់របស់ជំពូក

រចនាសម្ព័ន្ធថត
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ថត
     - ខ្លឹមសារ
   * - ``01_main-chapter-code``
     - កូដមេរបស់ជំពូក
   * - ``02_bonus_additional-experiments``
     - ការពិសោធន៍បន្ថែម៖ finetune ស្រទាប់ផ្សេងៗ និងប្រើម៉ូដែលធំជាង
   * - ``03_bonus_imdb-classification``
     - Finetuning ម៉ូដែលផ្សេងៗលើ dataset រឿងភាគ IMDb 50k (50k IMDb Movie Review Dataset)
   * - ``04_user_interface``
     - ការសាងសង់ User Interface សម្រាប់ប្រើជាមួយ GPT-based Spam Classifier

លំហាត់
--------

ចម្លើយលំហាត់មានក្នុង ``exercise-solutions.ipynb`` ខាងលើ និងក្នុងសង្ខេបនៃភាគបន្ថែម C។

.. tip::

   បន្ទាប់៖ finetune ម៉ូដែលឱ្យធ្វើតាមការណែនាំដូច chatbot ក្នុង
   :doc:`ជំពូក៧-finetuning-instructions`។
