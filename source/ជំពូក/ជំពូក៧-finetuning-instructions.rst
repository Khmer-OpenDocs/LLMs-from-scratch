.. meta::
   :description: ជំពូក ៧ Finetuning ដើម្បីធ្វើតាមការណែនាំ — instruction dataset, Ollama, DPO និង preference tuning។ Chapter 7 Finetuning to Follow Instructions — instruction dataset, Ollama, DPO and preference tuning.
   :keywords: ជំពូក ៧, instruction finetuning, ollama_evaluate.py, Ollama, DPO, preference tuning, dataset generation, Llama 3.1

ជំពូក ៧៖ Finetuning ដើម្បីធ្វើតាមការណែនាំ
====================================================

.. note::

   ប្រភព៖ ថត `ch07 <https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07>`_
   និងតារាងមាតិកាក្នុង `README.md <https://github.com/rasbt/LLMs-from-scratch/blob/main/README.md>`_ របស់ upstream។

ជំពូកចុងក្រោយនៃជំពូកមេ — ការ finetune ម៉ូដែល GPT ឱ្យធ្វើតាមការណែនាំ
(instruction following) ដូចជាជំនួយការសន្ទនា។ បណ្ណាល័យដែល ``requirements.txt``
កំណត់សម្រាប់ជំពូកនេះ៖ ``matplotlib`` · ``tensorflow`` · ``tqdm`` · ``psutil``។

កូដមេ
-------

កូដមេស្ថិតក្នុងថត
`ch07/01_main-chapter-code <https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07/01_main-chapter-code>`_៖

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - ឯកសារ
     - តួនាទី
   * - `ch07.ipynb <https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ch07.ipynb>`_
     - notebook មេរបស់ជំពូក
   * - `gpt_instruction_finetuning.py <https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/gpt_instruction_finetuning.py>`_
     - សង្ខេប (summary) ស្គ្រីប instruction finetuning
   * - `ollama_evaluate.py <https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ollama_evaluate.py>`_
     - សង្ខេប (summary) ការវាយតម្លៃតាមរយៈ Ollama
   * - `exercise-solutions.ipynb <https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/exercise-solutions.ipynb>`_
     - ចម្លើយលំហាត់របស់ជំពូក

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
