# Segment Anything for Circuit Recognition?
## Big Picture
- Can we train a prompt encoder that uses the existing child boxes?
  - and parents?
- Overall algorithm:
  - select every point
  - segment using that point as input
  - classify each masked image
  - NMS
  - iteratively:
    - pass odd/even levels of hierarchy into prompt encoder
    - predict the others

## Misc Thoughts
- we might want a model that predicts prior based solely on relative location...
  - some of these cropped out regions don't make sense outside of the location (like Ports w/o any indicator)

## Initial Vetting
- Are the prompts expressive enough to be able to segment out the regions we want?
  - [ ] Can we perform gradient-based search on the prompts?
    - Given an image, try to learn a prompt for something like "resistors"
    - [x] run an arbitrary prompt embedding into the pipeline
      - plan of attack:
        - [ ] create custom Sam
          - override `predict`
          - [ ] hack something in for now
            - the main method is straight-forward to replace
            - positional encoding?
              - what exactly is this?
              - it is used as the second input to the transformer
              - what are the tokens?
                - embeddings (iou_token, mask_tokens)
        - SamPredictor (efficient multiple calls)
        - probably don't need to AMG
          - how does it use SamPredictor?
            - `predict_torch` with `in_points`, `in_labels`
        - we prob want to use sparse encodings

        - what should be the pe returned from the prompt encoder?
          - what are these things?
            - `image_embedding` (64, 256, 64, 64)
            - `image_pe` (64, 256, 64, 64)
              - how is this computed by the current prompt encoder?
                - PositionEmbeddingRandom
                - generates position information used when determining the attn for the image
                - can probably reuse this with our approach
            - `point_embedding` (64, 7, 256)

            - `query_pe`: summed with the query at each step
              - same as `query`
            - `image_pe`: summed with the image at each step

        - could we actually use `no_mask_embed` instead?
          - it appears so...
          - enable gradients for that layer only
          ```python
          with torch.set_grad_enabled(False):
            # Create the layers
            # Set requires_grad=True
          ```

    - [ ] make a forward pass and get the output in a tensor

    - [ ] set up the evaluation
      - what was the loss?
        - dice loss & focal loss (20:1)
      - do I have to follow the exact same loss function?
        - let's try something simpler to start

- Prompt embedding info:
  - how big are the prompt embeddings?
    - 256 dimensions
  - what is the difference btwn sparse & dense embeddings in the code?

## To Do
- [ ] how can we start validating things?
  - [ ] can a GM be used to refine

- [x] write up an overview of the approach
  - skip to the prompt model portion?

- [-] save the masks for the components
  - can I download them from the website?
  - maybe load it in an iframe? Or add a plugin?
    - iframe is no good due to sandbox
    - we could run our own but this would not be connected to the GPU
      - nice not to mess with that infra
      - maybe it isn't too bad. We should check...
  - salt saves them in coco format already
  - we should already have enough data. Can we just generate the embeddings from ours?
    - can we use flatten to crop them all out?
       - then pass them to the embedding script (next item)?
- [x] generate the embeddings
    - [x] check if we are missing "Port" labels
      - nope
      - error in the flatten script
    - [x] why don't I have images of just the ports?
    - [x] place them in directories of their own
      - [x] get the labels from the filenames
      - [x] create directories for each class
      - [x] move each file to the 
  - [x] write a custom script to just do this
    - [x] I might be able to use the script from the sam labeling tool...
  - [-] generate from the coco format?
    - or from the mask directly?

- [ ] Given the embeddings for the images, can we classify points?
  - [x] abstract, circuits, resistor/capacitor?
  - [ ] train multi-class SVM
    - what error do I get with this approach?
    - don't expect much here

  - [ ] train a CNN for image classification?
    - what would be a good architecture? Maybe something a little dated like Alexnet?
    - EfficientNetV2 looks promising
    - it would be nice if we didn't need to train only on ours...
      - what if we could produce an embedding as an output rather than the class?
      - has anyone tried to tie it to a word embedding or something?
      - this is a big challenge...
    - let's just try some network
      - small since it is operating on an embedding already (64x64)

    - [x] update lenet to...
      - [x] correct # of channels
      - [x] # of outputs

    - [ ] train!
      - [x] make the dataset again with the correct shape. I probably shouldn't have just deleted the other ones
      - [x] keep running out of memory
        - need to stream the data instead
        - split the files into individual directories
      - [x] Reaching high 80s for accuracy on training data. Let's do this with validation data, too
      - [ ] overfitting. Simplifying NN to start. Otherwise, I can also check out
        - [x] removed 1 conv layer
          - still overfitting
        - [x] removed a dense layer. Should be a linear classifier now
          - if this doesn't work, I will add regularization
          - looks like it is still overfitting (~50% on training so far and still around ~10 on val)
          - 0.55 by epoch 7
        - [ ] add regularization
          - [x] L2 reg everywhere
            - poor performance but doesn't appear to be overfitting...
          - [ ] L1 reg on dense layer
            - 001 -> ~0.20
            - 000_01 -> ~0.20 (error)
            - 0001 -> overfitting

- [ ] generate masks from every point
  - classify then NMS

- [ ] train prompt encoder
  - two options:
      - input: parents
      - output: children
    - or:
      - input: parents and children
      - output: self

  - use a graph embedding?
    - prompt encoder as a graph or node embedding task?
      - or node embedding? (+ index)
        - single message?
    - graph would be a containment graph (spatially)
      - relationship (contained, containing) concatenated with one-hot component type
    - how many messages to send?
      - send 1 message? Or perhaps more?
      - start with 1

  - define a drop-in replacement for the default PromptEncoder
    - just hack it in SAM
    - does it currently support training the prompt encoder only?
      - I don't see any training step but that is probably fine since they are all just pytorch modules...
