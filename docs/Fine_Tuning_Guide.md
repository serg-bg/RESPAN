# Fine-Tuning a RESPAN Segmentation Model

> **Who this is for.** You've run RESPAN on your data, and in some images the model misses spines, adds spurious ones, or segments dendrites incorrectly. This guide walks you through **fine-tuning** our pretrained nnU-Net on a small set of your own corrected images, so the model learns your specific conditions (microscope, labeling, cell type) without you having to annotate from scratch.
>
> **Why fine-tune instead of train from scratch.** Training nnU-Net from scratch needs many annotated volumes and >24 hours on a GPU. Fine-tuning needs **as few as 5–10 corrected volumes** and can train in less time. You start from our pretrained weights and fine-tune them toward your data.
>
> This does not require access to the images we used to train our model. Fine-tuning works with the model file alone.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Big picture](#big-picture)
3. [Step 1: Run RESPAN on a handful of your images](#step-1-run-respan-on-a-handful-of-your-images)
4. [Step 2: Correct RESPAN's segmentation outputs](#step-2-correct-respans-segmentation-outputs)
5. [Step 3: Set up the folder structure nnU-Net expects](#step-3-set-up-the-folder-structure-nnu-net-expects)
6. [Step 4: Create your `dataset.json`](#step-4-create-your-datasetjson)
7. [Step 5: Get the RESPAN pretrained model bundle](#step-5-get-the-respan-pretrained-model-bundle)
8. [Step 6: Tell nnU-Net where its three folders live](#step-6-tell-nnu-net-where-its-three-folders-live)
9. [Step 7: Copy RESPAN's plans into your dataset](#step-7-copy-respans-plans-into-your-dataset)
10. [Step 8: Preprocess your dataset](#step-8-preprocess-your-dataset)
11. [Step 9: Fine-tune the model](#step-9-fine-tune-the-model)
12. [Step 10: Use your fine-tuned model in RESPAN](#step-10-use-your-fine-tuned-model-in-respan)
13. [How long does it take?](#how-long-does-it-take)
14. [Common problems and fixes](#common-problems-and-fixes)
15. [FAQ](#faq)

---

## Prerequisites

Before you start, confirm you have:

- [ ] **An NVIDIA GPU** with at least 12 GB of VRAM (same requirement as running RESPAN).
- [ ] **RESPAN installed** (either the Windows application or the Python environment from the main README).
- [ ] **A working nnU-Net install.** If you're running RESPAN from source, you already have the `respan_nnunet` conda environment, that includes nnU-Net v2. If you're running the Windows `.exe`, you'll need to install nnU-Net separately into a conda environment (see the main README's "Advanced usage" section for the one-line install).
- [ ] **5–20 images** from your own data that RESPAN has already analyzed, covering the problem cases you want the model to handle better.
- [ ] **An annotation tool.** We recommend **Fiji** (ImageJ) since you may already use it for microscopy. [Napari](https://napari.org) and [ITK-SNAP](http://www.itksnap.org) are also suitible.
- [ ] **About 10–20 GB of free disk space** for the nnU-Net working folders.

Throughout this guide we'll use this running example:

| Placeholder | Example value |
|---|---|
| RESPAN pretrained model | **Dataset219_Model1Bv2** (our current recommended model) |
| Your fine-tuning dataset | **Dataset701_MySpineFineTune** |
| Your working folder | `D:\RESPAN_Finetune\` |
| Configuration | `3d_fullres` (the only config RESPAN uses) |
| Fold | `all` (RESPAN runs inference with `-f all`) |

**Important:** Dataset IDs are just numbers. Ours use 200-series (217, 219, etc.), pick anything in the 500–999 range for your dataset so it can't collide with ours. The rest of the name (`_MySpineFineTune`) is a free-form label you choose.

---

## Big picture

Here's the whole workflow at a glance. Each step has its own section below.

```
┌──────────────────────────────────────────────────────────────────┐
│  1. Run RESPAN on 5-20 of your images                            │
│                                                                  │
│  2. Open each RESPAN segmentation in Fiji/Napari and correct it  │
│     (fix missed spines, remove false positives, clean edges)     │
│                                                                  │
│  3. Arrange your images + corrected labels into nnU-Net's        │
│     folder layout (imagesTr/, labelsTr/)                         │
│                                                                  │
│  4. Write a small dataset.json file describing your data         │
│                                                                  │
│  5. Download the RESPAN pretrained model bundle                  │
│                                                                  │
│  6. Set three Windows environment variables                      │
│     (nnUNet_raw, nnUNet_preprocessed, nnUNet_results)            │
│                                                                  │
│  7. Copy RESPAN's plans file into your dataset's preprocessed    │
│     folder (uses nnUNetv2_move_plans_between_datasets)           │
│                                                                  │
│  8. Run nnUNetv2_preprocess to convert your TIFs to the          │
│     format nnU-Net trains on                                     │
│                                                                  │
│  9. Run nnUNetv2_train with -pretrained_weights pointing to      │
│     RESPAN's checkpoint, this is the actual fine-tuning         │
│                                                                  │
│ 10. Point RESPAN at your newly trained model and run your data   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Run RESPAN on a handful of your images

Before you can correct RESPAN's outputs, you need outputs to correct. Run RESPAN normally on 5–20 representative images, ideally the ones where you've noticed the model struggling.

After the run, RESPAN creates a `Validation_Data/Segmentation_Labels/` folder inside your data directory. That folder contains one `.tif` per input image, and each TIF is the segmentation mask using this label scheme (for **Model 1Bv2**, the current recommended model):

| Label value | Structure |
|---|---|
| 0 | Background |
| 1 | Spine core |
| 2 | Spine shell|
| 3 | Dendrite |
| 4 | Axon |
| 5 | Soma |

If you used a different RESPAN model (Model 1A, 2, or 3), check the "Segmentation Model Types" table in the main `README.md` for that model's label scheme. **The label values you correct in Step 2 must exactly match the scheme your pretrained model uses:** the whole point of fine-tuning is that the new data teaches the same labels better, not different labels.

---

## Step 2: Correct RESPAN's segmentation outputs

This is where you teach the model what "right" looks like.

For each of your 5–20 images, open the original image and RESPAN's segmentation side-by-side, and fix mistakes you see:

- **Missed structures**, add them with the paint/brush tool using the correct label value.
- **Spurious detections**, paint over them with label 0 (background).
- **Bad edges**, tidy the boundaries where RESPAN's edges are clearly wrong.

**In Fiji (recommended for most users):**

1. `File → Open` your original `.tif` image.
2. `File → Open` the corresponding file from `Validation_Data/Segmentation_Labels/`.
3. Enable `Image → Color → Channels Tool... → Color`. Set the look-up-table on the segmentation so each label has a distinct color (`Image → Lookup Tables → glasbey` works well).
4. Open the `Brush Tool` (double-click it to set brush width) or the `Pencil Tool`.
5. To paint a specific label value, set it as the **foreground color** using `Edit → Options → Colors...`, use "White" = 255 as a temporary, then use `Process → Math → Replace...` to swap 255 → your target label (e.g. 1 for spine core). Alternatively, open `Edit → Selection → Create Selection` → `Edit → Fill` with the label value.
6. Scroll through Z one slice at a time and fix what you see.
7. `File → Save As → Tiff...`, **save as a 16-bit unsigned TIFF** (or 8-bit if the label count is small, either works as long as it's unsigned integer, not float).


We also provide imageJ macros for converted labels to channels and then back to labels. These make editing annotations easier.

**In Napari:**

```python
import napari, tifffile
viewer = napari.Viewer()
viewer.add_image(tifffile.imread('my_image.tif'))
labels = viewer.add_labels(tifffile.imread('my_image_segmentation.tif'))
napari.run()
# Edit with the brush/eraser, then save from the viewer's labels layer:
tifffile.imwrite('my_image_corrected.tif', labels.data.astype('uint8'))
```

**Quality tips:**

- Focus on **systematic mistakes**, if RESPAN consistently misses one spine class, that's what fine-tuning will fix.
- Do **not** mix label schemes. If RESPAN's output used `1=spine core, 2=spine shell, 3=dendrite`, your corrected file must use the same integers.
- Scaling is important, and in the current version of RESPAN we have set scalling to 1,1,1 and handle scaling internally, to ensure better handling of scaling for large datasets. This present a challenge for fine tuning as nnunet can't autoscale your data to match model. To address this, after annotating, it will best to rescale the raw and label data to match the native resolution of the model. Avoid interpolating or averaging when scaling label data as it will alter the label values.

At the end of Step 2 you should have a set of paired files like:

```
my_corrections/
├── case_001_image.tif        ← original image
├── case_001_labels.tif       ← corrected segmentation
├── case_002_image.tif
├── case_002_labels.tif
├── ...
```

---

## Step 3: Set up the folder structure nnU-Net expects

nnU-Net is strict about folder layout and file naming. Pick a working folder, for example `D:\RESPAN_Finetune\`, and create these three sub-folders:

```
D:\RESPAN_Finetune\
├── nnUNet_raw\
├── nnUNet_preprocessed\
└── nnUNet_results\
```

Inside `nnUNet_raw\`, create a dataset folder with a specific name format: **`Dataset<ID>_<Name>`**. For our example:

```
D:\RESPAN_Finetune\nnUNet_raw\Dataset701_MySpineFineTune\
├── imagesTr\
└── labelsTr\
```

Now copy your paired files into those two folders, **renaming them** to nnU-Net's convention:

- **Images**: `<case_name>_0000.tif`:** the `_0000` suffix is the channel number** (all RESPAN inputs are single-channel fluorescence, so always `_0000`).
- **Labels**: `<case_name>.tif`, **no channel suffix**. The case name must match the image exactly.

After renaming, your folder looks like:

```
Dataset701_MySpineFineTune\
├── imagesTr\
│   ├── case_001_0000.tif
│   ├── case_002_0000.tif
│   ├── case_003_0000.tif
│   └── ... (5-20 files)
├── labelsTr\
│   ├── case_001.tif
│   ├── case_002.tif
│   ├── case_003.tif
│   └── ... (same number of files, matching names)
└── dataset.json              ← we'll create this next
```

**The `_0000` is required**, even for single-channel data. If you leave it off, nnU-Net will refuse to process the dataset.

---

## Step 4: Create your `dataset.json`

`dataset.json` is a tiny text file that tells nnU-Net what it's looking at. Create it inside the `Dataset701_MySpineFineTune\` folder (next to `imagesTr\` and `labelsTr\`).

Here's a copy-paste template for **Model 1Bv2's label scheme** (spine core + shell + dendrite + axon + soma). Adjust the `numTraining` number to match how many cases you actually have.

```json
{
    "channel_names": {
        "0": "fluorescence"
    },
    "labels": {
        "background": 0,
        "spine_core": 1,
        "spine_shell": 2,
        "dendrite": 3,
        "axon": 4,
        "soma": 5
    },
    "numTraining": 10,
    "file_ending": ".tif"
}
```

**Key rules:**

- `channel_names` must have one entry per channel. Single-channel data has one entry with key `"0"`.
- `labels` must always include `"background": 0`. The rest must use consecutive integers (1, 2, 3, …) and the names must describe what that integer means in your label files. **They must match the pretrained model's scheme**, see the label table in Step 1.
- `numTraining` is the number of files in `imagesTr/` (i.e. the number of corrected cases you have).
- `file_ending` is `.tif` since your corrections are TIFFs. Do not use `.tiff`, nnU-Net treats the two as different formats.

**If you used Model 1A** (spines, dendrites, soma only, labels 1, 2, 3), replace the `labels` block with:

```json
    "labels": {
        "background": 0,
        "spine": 1,
        "dendrite": 2,
        "soma": 3
    },
```

**If you used Model 3** (two-photon in vivo, spines and dendrites only, labels 1, 2):

```json
    "labels": {
        "background": 0,
        "spine": 1,
        "dendrite": 2
    },
```

Save the file. **Do not add trailing commas or comments**. JSON is picky.

---

## Step 5: Get the RESPAN pretrained model bundle

From the [main README Segmentation Models table](../README.md#-pre-trained-segmentation-models), use the Google Form link to request access to the pretrained weights. After the form, you'll get a download link to a zip containing:

- `fold_all/checkpoint_final.pth` (or similar), **this is the pretrained weights file**
- `plans.json`, the preprocessing and architecture spec
- `dataset.json`, metadata from our training dataset

Unzip this bundle somewhere, e.g. `D:\RESPAN_Finetune\respan_pretrained\Dataset219_Model1Bv2\`. The unpacked layout should look like:

```
D:\RESPAN_Finetune\respan_pretrained\Dataset219_Model1Bv2\
├── nnUNetTrainer__nnUNetPlans__3d_fullres\
│   ├── fold_all\
│   │   ├── checkpoint_final.pth     ← this is the weights file
│   │   ├── checkpoint_best.pth
│   │   └── ...
│   ├── plans.json
│   └── dataset.json
└── ...
```

**If your bundle is missing `plans.json` or `dataset.json`, please contact us**, we need these to ship a fine-tunable model. (We are working on a bundle verification for future releases.)

---

## Step 6: Tell nnU-Net where its three folders live

nnU-Net reads three environment variables on startup. You have to set these **every time you open a new command prompt** before running nnU-Net commands (or set them permanently via Windows `System Properties → Environment Variables`).

**In a new Windows Command Prompt:**

```cmd
set nnUNet_raw=D:\RESPAN_Finetune\nnUNet_raw
set nnUNet_preprocessed=D:\RESPAN_Finetune\nnUNet_preprocessed
set nnUNet_results=D:\RESPAN_Finetune\nnUNet_results
```

**In PowerShell:**

```powershell
$env:nnUNet_raw = "D:\RESPAN_Finetune\nnUNet_raw"
$env:nnUNet_preprocessed = "D:\RESPAN_Finetune\nnUNet_preprocessed"
$env:nnUNet_results = "D:\RESPAN_Finetune\nnUNet_results"
```

**In Linux / macOS / WSL:**

```bash
export nnUNet_raw="/path/to/RESPAN_Finetune/nnUNet_raw"
export nnUNet_preprocessed="/path/to/RESPAN_Finetune/nnUNet_preprocessed"
export nnUNet_results="/path/to/RESPAN_Finetune/nnUNet_results"
```

Verify with `echo %nnUNet_raw%` (or `$env:nnUNet_raw` / `$nnUNet_raw`), it should print the path you just set.

Now activate your nnU-Net conda environment (created during RESPAN install):

```cmd
conda activate respan_nnunet
```

The commands in the remaining steps only work from inside this activated environment with the three variables set.

---

## Step 7: Copy RESPAN's plans into your dataset

nnU-Net stores preprocessing rules in a **plans file**. For fine-tuning to work, **your data has to be preprocessed with the same plans as the pretrained model**. We'll transfer RESPAN's plans into your dataset in two commands.

First, place the RESPAN bundle into `nnUNet_results\` (so nnU-Net can find it) and extract a fingerprint of your dataset:

```cmd
xcopy /E /I "D:\RESPAN_Finetune\respan_pretrained\Dataset219_Model1Bv2" "D:\RESPAN_Finetune\nnUNet_results\Dataset219_Model1Bv2"

nnUNetv2_extract_fingerprint -d 701
```

What this does: `xcopy` duplicates the pretrained bundle into `nnUNet_results\` where nnU-Net expects trained models. The `extract_fingerprint` command scans your 5–20 corrected images and records their shape, spacing, and intensity statistics into `nnUNet_preprocessed\Dataset701_MySpineFineTune\dataset_fingerprint.json`.

**You also need RESPAN's plans file in a matching fingerprint-accessible location.** Copy our `plans.json` (from inside the bundle) into `nnUNet_preprocessed\Dataset219_Model1Bv2\nnUNetPlans.json`:

```cmd
mkdir "D:\RESPAN_Finetune\nnUNet_preprocessed\Dataset219_Model1Bv2"
copy "D:\RESPAN_Finetune\respan_pretrained\Dataset219_Model1Bv2\plans.json" "D:\RESPAN_Finetune\nnUNet_preprocessed\Dataset219_Model1Bv2\nnUNetPlans.json"
```

Now run the transfer:

```cmd
nnUNetv2_move_plans_between_datasets -s 219 -t 701 -sp nnUNetPlans -tp nnUNetPlans
```

**Flag explanation:**

- `-s 219`, **source** dataset ID (where plans come from → RESPAN's Dataset219)
- `-t 701`, **target** dataset ID (where plans go to → your Dataset701)
- `-sp nnUNetPlans`, source plans identifier (the name of the plans JSON we just copied, without `.json`)
- `-tp nnUNetPlans`, target plans identifier (the name to save plans as in your dataset)

After this command, `nnUNet_preprocessed\Dataset701_MySpineFineTune\nnUNetPlans.json` will exist and contain RESPAN's preprocessing rules adapted to your dataset.

---

## Step 8: Preprocess your dataset

This step applies the transferred plans to your actual image files, resampling, normalizing intensity, and saving them in nnU-Net's fast-loading internal format:

```cmd
nnUNetv2_preprocess -d 701 -plans_name nnUNetPlans -c 3d_fullres --verify_dataset_integrity
```

**Flag explanation:**

- `-d 701`, your dataset ID.
- `-plans_name nnUNetPlans`, use the plans file we just created (matches the `-tp` from Step 7).
- `-c 3d_fullres`, configure for 3D full-resolution. This is the only configuration RESPAN uses.
- `--verify_dataset_integrity`, check that image shapes, label values, and case counts are consistent. Highly recommended on a first run.

This step takes 5–20 minutes depending on how many images you have and their size. If it fails, see [Common problems and fixes](#common-problems-and-fixes), almost all first-time errors happen here.

At the end, you'll have a populated `nnUNet_preprocessed\Dataset701_MySpineFineTune\nnUNetPlans_3d_fullres\` folder ready for training.

---

## Step 9: Fine-tune the model

### What fine-tuning actually does

Before you run the command, it's worth understanding what "fine-tuning" means mechanically, so you can make good choices in the next sections.

When nnU-Net loads RESPAN's pretrained weights, it does this (verified from the nnU-Net v2 source):

- **Encoder (feature extractor): fully transferred.** Every convolutional block that learned "what a spine looks like, what a dendrite edge looks like" comes directly from our pretrained model. No randomization.
- **Decoder (upsampling path): fully transferred.** The skip-connection blocks that reconstruct spatial detail also come from our model.
- **Segmentation head (final output layers): randomly re-initialized.** nnU-Net always skips the final classification layers (keys containing `.seg_layers.`) from the checkpoint. This is by design.
- **Strict shape matching elsewhere.** Every other layer must have the exact same shape as in our pretrained model, which is guaranteed by the plans transfer you did in Step 7.

So you are **not** training from scratch. You start with ~95% of the model's learned representation intact. Fine-tuning nudges it toward your data. The only "from scratch" piece is the final layer that maps rich features to label numbers, and that layer learns quickly because everything upstream is already good.

### Risk of deviating from the pretrained behavior

Three things happen during fine-tuning:

1. **Early epochs (0–50): loss spikes.** The fresh segmentation head can't yet read the transferred features, so training loss looks bad for the first ~50 epochs. **This is normal**, don't panic.
2. **Mid epochs (50–500): the seg head aligns and the encoder drifts slightly.** Gradients flow from the fresh seg head back into the encoder, which shifts encoder features a little to suit your data better.
3. **Late epochs (500+): overfitting risk.** With only 5–20 fine-tuning cases, long training eventually memorizes them. The encoder may lose general competence on data that looks like our original training set (catastrophic forgetting).

**How much risk in practice?**

- **LOW** if your data is similar to our training (same microscope class, similar resolution, similar spine morphology, most users correcting small error modes).
- **MODERATE** if your data is very different (a new modality, resolution, or cell type the pretrained model wasn't exposed to).

### How many epochs?

The nnU-Net **default is 1000 epochs** (250 iterations per epoch, 250,000 minibatches total). **That default is calibrated for training from scratch on a large dataset:** it's too many for fine-tuning on a handful of cases, and you will overfit.

Use this table instead:

| Your dataset size | Suggested stop point | Reason |
|---|---|---|
| 5–10 cases | **150–300 epochs** | Minimal data; stop early to avoid overfitting |
| 10–20 cases | **300–500 epochs** | Enough data for a real shift |
| 20+ cases | **500–800 epochs** | Can push harder; still below the 1000 default |

### Running the command

```cmd
nnUNetv2_train 701 3d_fullres all -pretrained_weights "D:\RESPAN_Finetune\nnUNet_results\Dataset219_Model1Bv2\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_all\checkpoint_final.pth"
```

**Argument explanation (positional, then flag):**

- `701`, your dataset ID.
- `3d_fullres`, configuration (same as Step 8).
- `all`, fold identifier. **Use `all`** (not 0–4) because RESPAN runs inference with `-f all`. Training a single `all` fold uses all your data for training (no held-out validation split), the right choice when you only have 5–20 cases.
- `-pretrained_weights "..."`, path to the **RESPAN checkpoint `.pth` file**. The example path assumes you unpacked to the location shown in Step 5.

**On startup, watch for this line in the log:**

```
### Using pretrained weights: D:\...\checkpoint_final.pth
### ...loaded successfully
```

If you don't see a "loaded successfully" line, the pretrained weights did NOT apply and you're training from scratch. Stop and re-check the `-pretrained_weights` path.

### Stopping training at the right time (the key step)

nnU-Net saves a checkpoint **after every epoch** into `nnUNet_results\Dataset701_MySpineFineTune\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_all\checkpoint_latest.pth`. This means you can safely stop training at any point with **Ctrl+C** and still have a working model.

**Recommended procedure:**

1. Start the training command above and let it run.
2. Watch the training log (`progress.png` is auto-generated in the fold folder, opens in any image viewer and shows the loss curve).
3. When you reach your target epoch number (from the table above), press **Ctrl+C** in the training window.
4. **Promote the checkpoint.** nnU-Net reserves `checkpoint_final.pth` for a 1000-epoch completion. To use your early-stopped model, rename:
   ```cmd
   copy "D:\RESPAN_Finetune\nnUNet_results\Dataset701_MySpineFineTune\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_all\checkpoint_latest.pth" "D:\RESPAN_Finetune\nnUNet_results\Dataset701_MySpineFineTune\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_all\checkpoint_final.pth"
   ```
   RESPAN looks for `checkpoint_final.pth` at inference time, so this promotion step is required if you Ctrl+C'd early.

### How to tell if fine-tuning is helping or hurting

Because you're using `fold=all`, there's no automatic validation. You have to check manually:

1. **Hold out 2–3 images** before Step 3, don't include them in `imagesTr/`. Keep them aside as a small test set.
2. **After each candidate stop point** (e.g. epoch 150, 300, 500), run RESPAN with your fine-tuned model on the held-out images.
3. **Compare outputs visually against your expectations.** Improved on the error modes you corrected? Still good on the easy cases? Pick that checkpoint.
4. **If outputs get worse on easy cases as training progresses**, that's catastrophic forgetting, use the earlier checkpoint.

### Expected runtime

On a modern GPU:

| Hardware | Epochs | Wall time |
|---|---|---|
| RTX 4090 (24 GB) | 300 | ~60 min |
| RTX 4090 | 500 | ~100 min |
| RTX 3070 (8 GB) | 300 | ~2–3 hours |
| Titan RTX (24 GB) | 300 | ~90 min |

### Final folder contents

At the end, your `nnUNet_results\Dataset701_MySpineFineTune\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_all\` looks like:

```
├── checkpoint_final.pth       ← use this in RESPAN (may be the file you copied from checkpoint_latest.pth)
├── checkpoint_latest.pth      ← saved after every epoch
├── checkpoint_best.pth        ← best by training loss (not great with fold=all; prefer checkpoint_latest.pth)
├── plans.json
├── dataset.json
├── progress.png               ← loss curve
└── training_log.txt
```

---

## Step 10: Use your fine-tuned model in RESPAN

Your new model lives in `nnUNet_results\Dataset701_MySpineFineTune\`. To use it in RESPAN:

**If using the RESPAN GUI:** in the Analysis tab, browse to your new `Dataset701_MySpineFineTune\` folder as the segmentation model. Save and run.

**If using `Analysis_Settings.yaml` (batch mode):** Edit the file in each dataset folder and point the segmentation model path at your new Dataset folder. RESPAN infers the dataset ID (`701`) from the folder name and uses `fold_all`.

**Sanity-check the first run:** process one or two validation images, ones that were problematic *before* fine-tuning but that you did NOT use for training, and compare the new outputs to the old. You should see the specific error modes you corrected in Step 2 are now resolved.

---

## How long does it take?

End-to-end, with 10 corrected images:

| Step | Time |
|---|---|
| Step 1: Run RESPAN | 5–30 min (depends on image size) |
| Step 2: Corrections in Fiji/Napari | 1–4 hours (the slow part, scales with image count) |
| Steps 3–4: Folder setup + dataset.json | 20–30 min |
| Steps 5–6: Model download + env setup | 30–60 min |
| Steps 7–8: Plans transfer + preprocess | 10–30 min |
| Step 9: Training | 2–6 hours (unattended) |
| Step 10: First verification | 10 min |

**Total hands-on time: 3–6 hours. Total elapsed time: 6–12 hours** (most of the training runs overnight).

---

## Common problems and fixes

**`nnUNetv2_extract_fingerprint` complains about mismatched shapes between images and labels.**
→ Step 2's corrected label file must have exactly the same Z/Y/X dimensions as the original image. Re-check in Fiji with `Image → Properties...`.

**`file_ending` error during preprocess.**
→ nnU-Net v2 treats `.tif` and `.tiff` as different formats. Make sure `dataset.json` says `".tif"` (three letters) and your files end in `.tif`. Rename if needed.

**"Label 5 found in labelsTr but not declared in dataset.json."**
→ A stray non-zero value in your label file that's not in the `labels` block. Open the label TIF, run `Image → Adjust → Threshold` to spot unexpected values, and paint them to the correct label or 0.

**"Plans file not found" during `nnUNetv2_move_plans_between_datasets`.**
→ You skipped or mistyped the `copy` command in Step 7. Confirm the plans file exists at `nnUNet_preprocessed\Dataset219_Model1Bv2\nnUNetPlans.json`.

**Training fails with "size mismatch for weights" when loading `-pretrained_weights`.**
→ Your plans don't match the pretrained model's plans. Re-check that Step 7 transferred plans from Dataset219 (source) to Dataset701 (target), not the other way around.

**GPU runs out of memory during training.**
→ Your plans inherited RESPAN's patch size (tuned for 24 GB GPUs). If you're on 12 GB, edit `nnUNet_preprocessed\Dataset701_MySpineFineTune\nnUNetPlans.json` and halve the `patch_size` values. You'll need to re-run Step 8's preprocess after editing.

**"Dataset 219 not found" during move_plans.**
→ You forgot the `xcopy` in Step 7 that placed the RESPAN bundle into `nnUNet_results\`. nnU-Net needs to see a Dataset219 folder to accept it as a source.

**Training seems stuck (loss not decreasing).**
→ 10–50 epochs is the warm-up period for fine-tuning. If the loss hasn't dropped by epoch 100, check that `-pretrained_weights` actually pointed to a valid `.pth` file (not a folder). The training log prints "Loading pretrained weights..." near the top if the load succeeded.

---

## FAQ

**Q: How many corrected images do I need?**
A: 5 at minimum; 10–20 should provide clear improvement

**Q: Can I fine-tune from Model 1A and use it on Model 1Bv2 label scheme (or vice versa)?**
A: No. The label scheme is encoded into the model's output layer. Fine-tune from the RESPAN model whose label scheme matches the data you're correcting.

**Q: Will fine-tuning make the model worse on the original RESPAN training data?**
A: Possibly yes, fine-tuning trades generality for specificity. Keep the original Dataset219 bundle around in case you want to switch back. The two models can coexist side-by-side in `nnUNet_results\`.

---

## Questions or issues?

File an issue on the [RESPAN GitHub repo](https://github.com/lahammond/RESPAN/issues) or contact the authors. We're actively iterating on this workflow and your feedback shapes the next version of this guide.
