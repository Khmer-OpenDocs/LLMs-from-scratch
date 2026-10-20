.. meta::
   :description: ជំពូក ៥ Pretraining លើទិន្នន័យគ្មានស្លាក — training loop, weight loading, learning rate schedulers និងស្ថាបត្យកម្ម LLM ផ្សេងៗ។ Chapter 5 Pretraining on Unlabeled Data — training loop, weight loading, schedulers and LLM architectures.
   :keywords: ជំពូក ៥, pretraining, gpt_train.py, gpt_generate.py, Gutenberg, learning rate scheduler, Llama, Qwen3, Gemma, Olmo, Aya, hyperparameter, tiktoken

ជំពូក ៥៖ Pretraining លើទិន្នន័យគ្មានស្លាក
=================================================

.. note::

   ប្រភព៖ ថត `ch05 <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch05>`_
   និងតារាងមាតិកាក្នុង `README.md <https://github.com/rasbt/LLMs-from-scratch/blob/main/README.md>`_ របស់ upstream។

ជំពូកនេះគ្របដណ្តប់ការបណ្តុះបណ្តាលមុន (pretraining) ម៉ូដែល GPT ដែលបានសាងសង់ក្នុងជំពូក ៤
លើទិន្នន័យគ្មានស្លាក (unlabeled data) រួមទាំងការបង្កើតបំណែកកូដសម្រាប់ train និង generate។
នេះជាជំពូកដែលមានថតបន្ថែមច្រើនជាងគេក្នុងឃ្លាំង — រួមទាំងការអនុវត្តស្ថាបត្យកម្ម LLM
ជំនាន់ថ្មីៗពីមូលដ្ឋាន។ បណ្ណាល័យដែល ``requirements.txt`` កំណត់សម្រាប់ជំពូកនេះ៖
``tiktoken`` · ``tensorflow`` · ``tqdm``។

កូដមេ
-------

កូដមេស្ថិតក្នុងថត
`ch05/01_main-chapter-code <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch05/01_main-chapter-code>`_៖

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - ឯកសារ
     - តួនាទី
   * - `ch05.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៥-pretraining.ipynb>`_
     - notebook មេរបស់ជំពូក
   * - `gpt_train.py <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/កូដ/ch05/01_main-chapter-code/gpt_train.py>`_
     - សង្ខេប (summary) ស្គ្រីបបណ្តុះបណ្តាល
   * - `gpt_generate.py <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/កូដ/ch05/01_main-chapter-code/gpt_generate.py>`_
     - សង្ខេប (summary) ស្គ្រីបបង្កើតអត្ថបទ
   * - `exercise-solutions.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៥-ចម្លើយលំហាត់.ipynb>`_
     - ចម្លើយលំហាត់របស់ជំពូក

Notebooks ខ្មែរសម្រាប់ជំពូកនេះ
-----------------------------

ឃ្លាំងឯកសារនេះមានកំណែ notebook ជាភាសាខ្មែរ (ថត ``ណូតប៊ុក-ខ្មែរ/``)៖

- `ជំពូក៥-pretraining.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៥-pretraining.ipynb>`_ — notebook មេ ព្រមទាំងការពន្យល់ និង comments ជាខ្មែរ
- `ជំពូក៥-ចម្លើយលំហាត់.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៥-ចម្លើយលំហាត់.ipynb>`_ — ចម្លើយលំហាត់របស់ជំពូក

សូមមើលបញ្ជីពេញលេញ និងរបៀបដំណើរការក្នុងទំព័រ :doc:`../យោង/ណូតប៊ុក-ខ្មែរ`។

សម្ភារៈបន្ថែម
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ថត
     - ខ្លឹមសារ
   * - ``02_alternative_weight_loading``
     - វិធីសាស្ត្រផ្ទុក weights ជំនួស
   * - ``03_bonus_pretraining_on_gutenberg``
     - Pretraining GPT លើ dataset Project Gutenberg
   * - ``04_learning_rate_schedulers``
     - ការបន្ថែមមុខងារបន្ថែមទៅក្នុង training loop
   * - ``05_bonus_hparam_tuning``
     - ការកែលម្អ hyperparameters សម្រាប់ pretraining
   * - ``06_user_interface``
     - ការសាងសង់ User Interface សម្រាប់ប្រើជាមួយ LLM ដែលបាន pretrain
   * - ``07_gpt_to_llama``
     - ការបំលែង GPT ទៅជា Llama (រួមមាន ``standalone-llama32.ipynb`` សម្រាប់ Llama 3.2 From Scratch)
   * - ``08_memory_efficient_weight_loading``
     - ការផ្ទុក weights ដោយសន្សំសំចៃអង្គចងចាំ (``memory-efficient-state-dict.ipynb``)
   * - ``09_extending-tokenizers``
     - ការពង្រីក Tiktoken BPE Tokenizer ដោយ token ថ្មី (``extend-tiktoken.ipynb``)
   * - ``10_llm-training-speed``
     - គន្លឹះប្រសិទ្ធភាព PyTorch សម្រាប់បង្កើនល្បឿនការ train LLM
   * - ``11_qwen3``
     - Qwen3 Dense និង Mixture-of-Experts (MoE) From Scratch
   * - ``12_gemma3``
     - Gemma 3 From Scratch
   * - ``13_olmo3``
     - Olmo 3 From Scratch
   * - ``14_ch05_with_other_llms``
     - ជំពូក ៥ ជាមួយ LLM ផ្សេងជា Drop-In Replacement (ឧ. Llama 3, Qwen 3)
   * - ``15_tiny-aya``
     - Tiny Aya From Scratch
   * - ``16_qwen3.5``
     - Qwen3.5 From Scratch
   * - ``17_gemma4``
     - Gemma 4 E2B និង E4B From Scratch
   * - ``18_muon``
     - សម្ភារៈបន្ថែមក្នុងឃ្លាំង (មិនមានក្នុងបញ្ជី ``README.md`` របស់ upstream)
   * - ``19_cpu_mps_differences``
     - ភាពខុសគ្នារវាងឧបករណ៍ CPU និង MPS

លំហាត់
--------

ចម្លើយលំហាត់មានក្នុង ``exercise-solutions.ipynb`` ខាងលើ និងក្នុងសង្ខេបនៃភាគបន្ថែម C។

.. tip::

   បន្ទាប់៖ ប្រើម៉ូដែលដែលបាន pretrain ដើម្បីចំណាត់ថ្នាក់អត្ថបទក្នុង
   :doc:`ជំពូក៦-finetuning-classification`។
