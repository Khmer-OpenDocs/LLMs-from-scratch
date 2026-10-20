.. meta::
   :description: ជំពូក ៦ Finetuning សម្រាប់ចំណាត់ថ្នាក់អត្ថបទ — spam classifier, dataset IMDb និង UI។ Chapter 6 Finetuning for Text Classification — spam classifier, IMDb dataset and UI.
   :keywords: ជំពូក ៦, finetuning, text classification, gpt_class_finetune.py, spam classifier, IMDb, pandas, matplotlib

ជំពូក ៦៖ Finetuning សម្រាប់ចំណាត់ថ្នាក់អត្ថបទ
======================================================

.. note::

   ប្រភព៖ ថត `ch06 <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch06>`_
   និងតារាងមាតិកាក្នុង `README.md <https://github.com/rasbt/LLMs-from-scratch/blob/main/README.md>`_ របស់ upstream។

ជំពូកនេះបង្ហាញការកែសម្រួល (finetune) ម៉ូដែល GPT ដែលបាន pretrain ហើយ
ឱ្យក្លាយជាម៉ូដែលចំណាត់ថ្នាក់អត្ថបទ (text classification)។ បណ្ណាល័យដែល
``requirements.txt`` កំណត់សម្រាប់ជំពូកនេះ៖ ``matplotlib`` · ``tensorflow`` · ``pandas``។

មេរៀនសង្ខេប (សិក្សាងាយ)
--------------------------

ការពន្យល់ខាងក្រោមជាអត្ថបទដើមរបស់សហគមន៍ ដើម្បីជួយឱ្យយល់គំនិតសំខាន់ៗមុនពេលអានកូដ៖

- **Transfer learning** — យកម៉ូដែលដែលរៀនរួចមកកែសម្រួលបន្តលើកិច្ចការថ្មី ដោយប្រើទិន្នន័យ និងពេលតិចជាងការរៀនពីសូន្យ។
- **Freeze / unfreeze** — ចាក់សោ weights ភាគច្រើន ហើយរៀនតែស្រទាប់ខ្លះ (ឬក្បាលចំណាត់ថ្នាក់) ដើម្បីកាត់បន្ថយ overfitting និងការគណនា។
- **Classification head** — ជំនួសស្រទាប់លទ្ធផល "ព្យាករណ៍ token បន្ទាប់" ដោយស្រទាប់ដែលបង្ហាញពិន្ទុចំណាត់ថ្នាក់ (ឧ. spam/ham)។
- **Balanced dataset** — ធ្វើឱ្យចំនួនគំរូក្នុងថ្នាក់នីមួយៗប្ហាក់ប្ហែលគ្នា ដើម្បីកុំឱ្យម៉ូដែលលំអៀងទៅថ្នាក់ធំ។
- **Accuracy** — សមាមាតនៃការព្យាករណ៍ត្រូវ។ ត្រូវតាមដានទាំង train និង validation។

ដំណើរការកូដក្នុងជំពូកនេះ៖

1. ទាញយក និងថ្លឹងថ្លែង dataset សារ spam/ham។
2. សាងសង់ SpamDataset និង loaders។
3. ផ្ទុក GPT-2 pretrained។
4. ចាក់សោ weights ជំនួសក្បាលចំណាត់ថ្នាក់ និងដោះសោប្លុកចុងក្រោយ។
5. សរសេរអនុគមន៍ accuracy/loss/train ហើយបណ្តុះបណ្តាល។
6. សាកល្បង classify_review និងរក្សាទុកម៉ូដែល។

កូដមេ
-------

កូដមេស្ថិតក្នុងថត
`ch06/01_main-chapter-code <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch06/01_main-chapter-code>`_៖

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - ឯកសារ
     - តួនាទី
   * - `ch06.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៦-finetuning-classification.ipynb>`_
     - notebook មេរបស់ជំពូក
   * - `gpt_class_finetune.py <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/កូដ/ch06/01_main-chapter-code/gpt_class_finetune.py>`_
     - ស្គ្រីប finetuning សម្រាប់ចំណាត់ថ្នាក់
   * - `exercise-solutions.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៦-ចម្លើយលំហាត់.ipynb>`_
     - ចម្លើយលំហាត់របស់ជំពូក

Notebooks ខ្មែរសម្រាប់ជំពូកនេះ
-----------------------------

ឃ្លាំងឯកសារនេះមានកំណែ notebook ជាភាសាខ្មែរ (ថត ``ណូតប៊ុក-ខ្មែរ/``)៖

- `ជំពូក៦-finetuning-classification.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៦-finetuning-classification.ipynb>`_ — notebook មេ ព្រមទាំងការពន្យល់ និង comments ជាខ្មែរ
- `ជំពូក៦-ចម្លើយលំហាត់.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៦-ចម្លើយលំហាត់.ipynb>`_ — ចម្លើយលំហាត់របស់ជំពូក

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
