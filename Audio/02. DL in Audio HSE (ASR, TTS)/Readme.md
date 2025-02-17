### О чем тут=)
В этой папочке я буду изучать курс [DL in Audio от HSE](https://github.com/TaliyIvanov/dlafromHSE/tree/2024)

Собственно здесь будут заметки, ссылки, файлики, которые идут по неделям, мне на будущее, 
чтобы я мог к ним всегда вернуться=)

### Week 01:
Вступительная часть. Рассказывают про  PyTorch. 
Для себя полезными выделил следующие вещи:
- [Hydra](https://hydra.cc/) - это фреймворк для управления файлами конфигурации, заточенный под ML-проекты.
- [Python Development Tips](https://pydevtips.readthedocs.io/en/latest/#pydevtips-python-development-tips) - про воспроизводимость кода
- [WandB](https://wandb.ai/site) -- easy to use and cool website to monitor logs from different machines (cloud-hosted)
- [Comet ML](https://www.comet.com/site/) -- similar to WandB with a bit smaller functionality (cloud-hosted)
- [MLflow](https://mlflow.org/) -- easy to use but UI is less intuitive and it is harder to compare logs from different machines (self-hosted)
- [TensorBoard](https://www.tensorflow.org/tensorboard) -- rather old but still popular, might consume a lot of resources (from seminar author's experience). However, it contains unique cool features like profiler support (see here). (self-hosted)
- [aim](https://github.com/aimhubio/aim) -- seminar author has not try it, but it is open-source which is cool (self-hosted)
- [ClearML](https://github.com/clearml/clearml) -- seminar author have not try it, but it contains tons of different features. Similar to WandB and Comet but free account has more restrictions (cloud-hosted).

### [Week 02](https://github.com/TaliyIvanov/dlafromHSE/tree/2024/week02):
Здесь идет повествования об основах звука, физики звука и базовых инструментах для работы с ним.
- [Waveform for dummies](https://pudding.cool/2018/02/waveforms/)
- [DFT and FFT](https://www.robots.ox.ac.uk/~sjrob/Teaching/SP/l7.pdf)
- Visualization of Nyquist-Shannon theorem
- [In details about MelScale and MFCC](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)
- [Fourier Transfor for dummies (YouTube)](https://www.youtube.com/watch?v=spUNpyF58BY&t=1s&ab_channel=3Blue1Brown)
- [FFT](https://en.wikipedia.org/wiki/Fast_Fourier_transform)
- [torch.fft - docs](https://pytorch.org/docs/stable/fft.html)
- [STFT](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)

### [Week 03 - Automatic Speech Recognition part - I](https://github.com/TaliyIvanov/dlafromHSE/tree/2024/week03)
Первая часть по Автоматическому распознованию Речи. Основы ASR
- [Lecture slides](https://docs.google.com/presentation/d/1cBXdNIbowwYNp42WhJmd1Pp85oeslOrKNmGyZa5HKBQ/edit?usp=sharing)
- [Sequence Modeling With CTC](https://distill.pub/2017/ctc/), a blog post explaining and visualizing CTC loss;
- [The original CTC Loss paper can be found here.](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- [Groups, Depthwise, and Depthwise-Separable Convolution, a video explaining why depthwise separable convolutions are needed and what their advantages are over regular convolutions.](https://www.youtube.com/watch?v=vVaRhZXovbw)
- [This tutorial from Torch shows how to use CTC Beam Search with language model support.](https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html)
- [Recording in English with a brief explanation of the CTC Loss can be found here.](https://youtu.be/YuImUy6vPFs)
- []()
- []()

### [Week 04 - Automatic Speech Recognition part - II] 
Вторая часть по Автоматическому распознования речи. Рассказывали про декодеры + немного ЛЛМ.

- [Начало семинара](https://youtu.be/4who1RG-kaA?t=8946)
- [1] [NeMo docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#hybrid-transducer-ctc)
- [2] [RNNT + LAS](https://arxiv.org/pdf/1908.10992)
- [3] [CTC + LAS](https://arxiv.org/pdf/1609.06773)
- [4] [Hybrid Rescoring 1](https://arxiv.org/pdf/2008.13093)
- [5] [Hybrid Rescoring 2](https://arxiv.org/pdf/2101.11577)