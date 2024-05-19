# func2stream

`func2stream` is a Python library that simplifies the construction of data processing pipelines, particularly for computationally intensive tasks such as real-time video processing and computer vision applications.

Inspired by Gstreamer's pipeline architecture, `func2stream` provides a clean and minimally invasive way to integrate your existing code into an efficient and scalable processing workflow.

## 🚧 Current Status and Roadmap

`func2stream` is currently in early development and is being actively worked on. While it shows promise, it is not yet recommended for production use. The following areas are being prioritized for improvement:

- [ ] 🎯 Strive towards a decorator-free solution

  - [x] Implement automatic conversion of SISO functions to Elements and connect them in `Pipeline([fn1, fn2...])`

    *Decorators are aesthetically unpleasing and may not provide a foolproof solution. As other projects have attempted and abandoned.*

  - [ ] Eliminate the need for decorators in MIMO functions
    - [ ] Introduce implicit Context construction using AST parsing
    - [ ] Simplify the process for users while maintaining necessary control and flexibility

- [ ] 🧩 Implement implicit Context

  - [ ] Extend the automation to MIMO functions
    - [ ] Use AST to parse function input and output parameters
    - [ ] Automatically convert parameters into variable keys
    - [ ] Enable users to build Context mode by ensuring consistent naming in function parameter lists and return statements
  - [ ] Simplify the process for users while maintaining necessary control and flexibility

## Key Features

### 1. Easy Integration with Existing Code

`func2stream` allows you to use your existing functions without modifying their internal logic. You can focus on your project's requirements while `func2stream` handles the underlying asynchronous `Pipeline` orchestration.

### 2. Intuitive Pipeline Construction

Building pipelines with `func2stream` is straightforward thanks to its `Element` abstraction and automatic `Pipeline` assembly.

- Functions can be easily transformed into `Element`s, promoting modularity and reusability in `Pipeline` design.
- Simply define the sequence of functions or Elements in your `Pipeline`, and `func2stream` will efficiently link them together.
- `Pipeline`s can be treated as `Element`s, allowing for the creation of complex, nested structures.

### 3. MapReduce Processing

`func2stream` incorporates the Map-Reduce paradigm, enabling easy parallelization of processing tasks for improved performance.

- Define concurrent `Element`s using the `MapReduce([fn1,fn2,fn3...])` syntax, and `func2stream` will handle parallel execution and result aggregation.
- The `MapReduce` functionality itself is an `Element`, seamlessly integrating parallel processing into your `Pipeline`.

### 4. Managed Context

`func2stream` introduces a managed `context` mechanism to streamline data flow through the pipeline, reducing the need for manual data management between Elements.

- The managed context handles data passing between Elements, making data usage across functions more intuitive and convenient.
- Decorators allow you to specify which variables an `Element` reads from and writes to the context.
- The centralized data store ensures that variables accessed and modified by different asynchronous steps remain independent, preventing unintended interactions and maintaining data integrity.
