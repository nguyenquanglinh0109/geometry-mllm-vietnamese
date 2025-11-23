# geometry-mllm-vietnamese

## Data collection
[Unigeo Dataset](https://drive.google.com/drive/folders/1NifdHLJe5U08u2Zb1sWL6C-8krpV2z2O?usp=share_link)

## Data preprocessing
### Math format
```
id: str
problem_type: str  # "calculation" or "proving"
question: str
image_path: Optional[str] = None
cot: Optional[str] = None

# For calculation
choices: Optional[List[str]] = None
answer: Optional[str] = None
choice: Optional[str] = None

# For proving
statements: Optional[List[str]] = None
reasons: Optional[List[str]] = None
elements: Optional[List[str]] = None
```

### Trasform symbol to text
#### Process english text

#### Process chinese text

#### Replace manual_program -> number (calculation data)