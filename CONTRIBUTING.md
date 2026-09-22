# Contributing

Keep changes small and preserve the experiment's three comparable arms.

1. Create a virtual environment and install `requirements.txt`.
2. Run `make test` and `make audit`.
3. Do not commit `.env`, API keys, local paths, or unsanitized run artifacts.
4. Describe any policy, prompt, model, pricing, or grading change that could affect results.

Paid benchmark runs are not required for ordinary pull requests.
