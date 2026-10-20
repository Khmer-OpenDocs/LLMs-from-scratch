.. meta::
   :description: ជំពូក ៧ Finetuning ដើម្បីធ្វើតាមការណែនាំ — instruction dataset, Ollama, DPO និង preference tuning។ Chapter 7 Finetuning to Follow Instructions — instruction dataset, Ollama, DPO and preference tuning.
   :keywords: ជំពូក ៧, instruction finetuning, ollama_evaluate.py, Ollama, DPO, preference tuning, dataset generation, Llama 3.1

ជំពូក ៧៖ Finetuning ដើម្បីធ្វើតាមការណែនាំ
====================================================

.. note::

   ប្រភព៖ ថត `ch07 <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch07>`_
   និងតារាងមាតិកាក្នុង `README.md <https://github.com/rasbt/LLMs-from-scratch/blob/main/README.md>`_ របស់ upstream។

ជំពូកចុងក្រោយនៃជំពូកមេ — ការ finetune ម៉ូដែល GPT ឱ្យធ្វើតាមការណែនាំ
(instruction following) ដូចជាជំនួយការសន្ទនា។ បណ្ណាល័យដែល ``requirements.txt``
កំណត់សម្រាប់ជំពូកនេះ៖ ``matplotlib`` · ``tensorflow`` · ``tqdm`` · ``psutil``។

មេរៀនសង្ខេប (សិក្សាងាយ)
--------------------------

ការពន្យល់ខាងក្រោមជាអត្ថបទដើមរបស់សហគមន៍ ដើម្បីជួយឱ្យយល់គំនិតសំខាន់ៗមុនពេលអានកូដ៖

- **Instruction dataset** — គំរូជា (instruction, input, output)។ ម៉ូដែលរៀន "ឆ្លើយតប" ជំនួសឱ្យការគ្រាន់តែបន្តអត្ថបទ។
- **Prompt template** — ការរៀបចំវាលទាំងនោះចូលជាខ្សែអក្សរតែមួយតាមទ្រង់ទ្រាយកំណត់មួយ។
- **Masking ដោយ -100** — គណនា loss តែលើផ្នែកចម្លើយ។ ផ្នែក instruction/input ត្រូវបានមិនអើពើ (ignore_index)។
- **Padding និង collate** — បំពេញបន្សំក្នុង batch ឱយមានបរវែងស្មើគ្នា ដើម្បីអាចគណនារួមគ្នាជាម៉ាទ្រីស។
- **ការវាយតម្លៃដោយម៉ូដែលផ្សេង** — ប្រើម៉ូដែលក្នុងមូលដ្ឋាន (Ollama) ពិនិត្យគុណភាពចម្លើយបន្ទាប់ពី finetuning។

ដំណើរការកូដក្នុងជំពូកនេះ៖

1. ផ្ទុក dataset JSON និងសង្កេតគំរូ។
2. សរសេរ format_input និងបំបែក train/val។
3. អភិវឌ្ឍ collate function ជាដំណាក់ (padding → masking)។
4. សាងសង់ InstructionDataset និង loaders។
5. ផ្ទុកម៉ូដែល pretrained ហើយ finetune លើការណែនាំ។
6. វាយតម្លៃចម្លើយដោយ Ollama។

កូដមេ
-------

កូដមេស្ថិតក្នុងថត
`ch07/01_main-chapter-code <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch07/01_main-chapter-code>`_៖

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - ឯកសារ
     - តួនាទី
   * - `ch07.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៧-finetuning-instructions.ipynb>`_
     - notebook មេរបស់ជំពូក
   * - `gpt_instruction_finetuning.py <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/កូដ/ch07/01_main-chapter-code/gpt_instruction_finetuning.py>`_
     - សង្ខេប (summary) ស្គ្រីប instruction finetuning
   * - `ollama_evaluate.py <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/កូដ/ch07/01_main-chapter-code/ollama_evaluate.py>`_
     - សង្ខេប (summary) ការវាយតម្លៃតាមរយៈ Ollama
   * - `exercise-solutions.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៧-ចម្លើយលំហាត់.ipynb>`_
     - ចម្លើយលំហាត់របស់ជំពូក

Notebooks ខ្មែរសម្រាប់ជំពូកនេះ
-----------------------------

ឃ្លាំងឯកសារនេះមានកំណែ notebook ជាភាសាខ្មែរ (ថត ``ណូតប៊ុក-ខ្មែរ/``)៖

- `ជំពូក៧-finetuning-instructions.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៧-finetuning-instructions.ipynb>`_ — notebook មេ ព្រមទាំងការពន្យល់ និង comments ជាខ្មែរ
- `ជំពូក៧-ចម្លើយលំហាត់.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៧-ចម្លើយលំហាត់.ipynb>`_ — ចម្លើយលំហាត់របស់ជំពូក

សូមមើលបញ្ជីពេញលេញ និងរបៀបដំណើរការក្នុងទំព័រ :doc:`../យោង/ណូតប៊ុក-ខ្មែរ`។

សម្ភារៈបន្ថែម
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ថត
     - ខ្លឹមសារ
   * - ``02_dataset-utilities``
     - ឧបករណ៍ dataset សម្រាប់រក near duplicates និងបង្កើត passive voice entries
   * - ``03_model-evaluation``
     - ការវាយតម្លៃចម្លើយ instruction ដោយប្រើ OpenAI API និង Ollama
   * - ``04_preference-tuning-with-dpo``
     - Direct Preference Optimization (DPO) សម្រាប់ LLM Alignment (``dpo-from-scratch.ipynb``) និងការបង្កើត Preference Dataset ជាមួយ Llama 3.1 70B និង Ollama (``create-preference-data-ollama.ipynb``)
   * - ``05_dataset-generation``
     - ការបង្កើត dataset សម្រាប់ instruction finetuning (``llama3-ollama.ipynb``) និងការកែលម្អ dataset (``reflection-gpt4.ipynb``)
   * - ``06_user_interface``
     - ការសាងសង់ User Interface សម្រាប់ប្រើជាមួយម៉ូដែល GPT ដែលបាន instruction-finetune

លំហាត់
--------

ចម្លើយលំហាត់មានក្នុង ``exercise-solutions.ipynb`` ខាងលើ និងក្នុងសង្ខេបនៃភាគបន្ថែម C។

.. tip::

   បន្ទាប់៖ សិក្សាភាគបន្ថែម A–E (PyTorch, LoRA ជាដើម) ក្នុងទំព័រ
   :doc:`../យោង/ភាគបន្ថែម`។
