### Setting Up and Downloading Baseline Model weights with Hugging Face

This guide outlines the steps to download (and check in) the models weights required for the baseline models.
We will focus on the `Meta-Llama-3-8B-Instruct` and `all-MiniLM-L6-v2` models.
But the steps should work equally well for any other models on hugging face. 

#### Preliminary Steps:

1. **Install the Hugging Face Hub Package**:
   
   Begin by installing the `huggingface_hub` package, which includes the `hf_transfer` utility, by running the following command in your terminal:

   ```bash
   pip install huggingface_hub[hf_transfer]
   ```

2. **Accept the Llama Terms**:
   
   You must accept the Llama model's terms of use by visiting: [meta-llama/Meta-Llama-3-8B-Instruct Terms](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).

3. **Create a Hugging Face CLI Token**:
   
   Generate a CLI token by navigating to: [Hugging Face Token Settings](https://huggingface.co/settings/tokens). You will need this token for authentication.

#### Hugging Face Authentication:

1. **Login via CLI**:
   
   Authenticate yourself with the Hugging Face CLI using the token created in the previous step. Run:

   ```bash
   huggingface-cli login
   ```

   When prompted, enter the token.

#### Model Downloads:

1. **Download LLaMA-2-7b Model**:

   Execute the following command to download the `Meta-Llama-3-8B-Instruct` model to a local subdirectory. This command excludes unnecessary files to save space:

   ```bash
   HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
       meta-llama/Meta-Llama-3-8B-Instruct \
       --local-dir-use-symlinks False \
       --local-dir models/meta-llama/Meta-Llama-3-8B-Instruct \
       --exclude *.pth # These are alternates to the safetensors hence not needed
   ```

3. **Download MiniLM-L6-v2 Model (for sentence embeddings)**:

   Similarly, download the `sentence-transformers/all-MiniLM-L6-v2` model using the following command:

   ```bash
   HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
      sentence-transformers/all-MiniLM-L6-v2 \
       --local-dir-use-symlinks False \
       --local-dir models/sentence-transformers/all-MiniLM-L6-v2 \
       --exclude *.bin *.h5 *.ot # These are alternates to the safetensors hence not needed
   ```
