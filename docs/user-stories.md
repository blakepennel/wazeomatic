# User stories

Points use the Fibonacci scale (1, 2, 3, 5, 8, 13, 21) and are estimated relative to the team's reference stories below, not in hours.

## Stories

| ID | Role | Story | Points | Sprint |
|----|------|-------|--------|--------|
| US1 | Driver | As a driver, I want potholes and work zones I pass to be reported to Waze without touching my phone, so that other drivers are warned and I stay focused on the road. | 13 | 1–2 |
| US2 | Driver | As a driver, I want the same hazard reported only once and flickering detections ignored, so that I do not spam Waze or lose report credibility. | 3 | 1 |
| US3 | Driver | As a driver, I want detection to run on my phone next to Waze without making navigation lag, so that I need no extra hardware. | 8 | 1–2 |
| US4 | Driver | As a driver, I want animals and stopped cars on the shoulder reported too, so that the less obvious hazards are covered. | 5 | 2 |
| US5 | Macro maintainer | As the teammate maintaining the macro, I want a fixed, documented Intent action and extras, so that pipeline changes do not break my macro. | 2 | 1 |
| US6 | Macro maintainer | As the macro maintainer, I want a DRY_RUN mode that does every step except send, so that I can test against real Waze without filing live reports. | 3 | 1 |
| US7 | Developer | As a developer, I want to replay dashcam clips through the pipeline on my PC with camera-style degradation, so that I can tune thresholds without driving. | 5 | 1 |
| US8 | Developer | As a developer, I want an emulator setup with a mock Waze app that runs the whole chain and checks which report was made, so that end-to-end tests are repeatable and safe. | 8 | 1–2 |
| US9 | Developer | As a developer, I want a hazard model trained from merged open datasets and exported to ONNX with benchmarks, so that we can pick a model size that fits the phone. | 13 | 1–2 |
| US10 | Pi user | As a driver with a Raspberry Pi camera, I want the Pi to send detections to my phone over Wi-Fi, so that the phone camera is free and the view is better. | 8 | 3 |
| US11 | Driver | As a driver, I want police sightings reported, so that the app covers Waze's most used category. | 21 | deferred |

## Estimation rationale

- **US1 (13):** compared to ref 1 (shell, 13) and ref 16 (backend integration, 13): many parts (camera, model, events, macro, Waze UI) built separately and debugged together on a real system.
- **US2 (3):** compared to ref 14 (concurrency control simulator, 3) and ref 3 (Nim, 3): clear rules (3 of 10 frames, cooldowns) in plain code with simple state, easy to unit test.
- **US3 (8):** compared to ref 4 (MCP server, 8) and ref 8 (Turing machine, 8): integrating an existing runtime (ONNX Runtime, XNNPACK) and debugging until it performs, with defined phases (pre-process, infer, decode).
- **US4 (5):** compared to ref 17 (SoilProve, 5) and ref 18 (BloodIQ, 5): animals come free from the COCO model, but the stopped-vehicle heuristic is a prototype needing domain tuning, like a hackathon prototype.
- **US5 (2):** compared to ref 9 (mock DMV, 2) and ref 2 (quicksort with pseudo code, 2): the schema already exists; it is writing down and freezing a small interface.
- **US6 (3):** compared to ref 3 (Nim, 3) and ref 21 (Makefile test suite, 3): a few extra branches in an existing macro plus a test procedure.
- **US7 (5):** compared to ref 6 (static personal website, 5) and ref 15 (website with update pipeline, 5): moderate tool with input handling and a few output modes, built on existing detector code.
- **US8 (8):** compared to ref 11 (Excel macro over variable-size exports, 8) and ref 4 (MCP server, 8): automation that has to cope with timing and UI variation, plus a second Android app to build.
- **US9 (13):** compared to ref 16 (backend with data processing and testing, 13) and below ref 5 (NL2SQL research, 21): dataset conversion, training and evaluation are substantial, but the model and tooling are standard, not novel research.
- **US10 (8):** compared to ref 4 (MCP server, 8) and ref 8 (Turing machine, 8): reuses the Python pipeline and the existing broadcast path; the new work is networking and running on unfamiliar hardware.
- **US11 (21):** compared to ref 5 (NL2SQL research, 21) and ref 7 (VM pipeline, 21): no usable dataset exists, so it needs data collection and labelling before any modelling, with an uncertain outcome.

## Reference stories

| ID | Points | Source | Summary |
|----|--------|--------|---------|
| 1 | 13 | EECS 678 S26 | Built a shell that could execute different commands like cd, echo, and pwd, as well as supporting jobs. Required knowledge of C and system calls. |
| 2 | 2 | EECS 330 S26 | Implemented the quick sort algorithm and the randomized quick sort algorithm for better average performance. Usage of python and provided psuedo code made task much easier. |
| 3 | 3 | EECS 348 FA25 | Built the game Nim using Haskell. Required handling of user input and game data types in pure function oriented language. Simple ruleset and game states made the task easier. |
| 4 | 8 | Personal Project | Built location platform MCP server to utilize synced location data in agentic sources. Integrated existing work that reverse engineered googles work and had to debug to get data ingestion fully working. |
| 5 | 21 | NL2SQL Research | Recreated baselines and papers to then create novel research experiments on applying vector quantized vision language action models to SQL code generation. Proved negative result and that free form standard tokenization is better. Lots of researching, function specifics, and data specifics needed. |
| 6 | 5 | Personal website | Built personal website with analytics. Required handling hosting, user viewing, input for clicked fields like buttons, etc. The nature of the site being a static site made things easier. |
| 7 | 21 | Internship | Built a gitlabs pipeline for creating and updating virtual machines on Nutanix clusters. Required accessing a large database of configuration item details, creating a JSON format for querying the gitlab pipeline, querying for existing virtual machine information from live infrastructure to ensure no double creation, and creating golden windows images. |
| 8 | 8 | EECS 510 RType Turing Machine | Built a turing machine in python that could calculate addition and subtraction using the MIPS Instruction Set architecture format. The machine was divided into three phases: compilation, decoding, and execution. Used a single python dictionary to contain every state, mimicing the functionality of real turing machines as closely as possible. |
| 9 | 2 | EECS 168 | Built a mock DMV using python. Allows for checking for license info from a database. Simple drivers license class implementation with basic sorting algorithms. Implemented in under an hour using basic programming skills |
| 10 | 21 | Internship | Built a mass phishing email deployment platform that allows for templated creation of phishing emails and conditional deployments to a sql based list of emails |
| 11 | 8 | Internship | Built a microsoft excel macro that ingests exported phishing data and outputs organized data statistics, ingested file would vary between a few hundred to a few thousand lines |
| 12 | 2 | Personal Project | Built a php site that allows for linux system command execution on an improperly configured website, it allowed for remote upload and shell creation and more |
| 13 | 1 | EECS 581 FA26 | Used AI to write a program to extract IPv4 Addresses from Noisy Text. This entailed coordinating AI to complete the task and correcting the errors of its output. |
| 14 | 3 | COSE 371 FA25 | Built a database concurrency control simulator using C++ which had the purpose of implementing two major concurrency control schemes, rigorous two-phase locking and opEmisEc concurrency control. |
| 15 | 5 | Personal website | Built a personal website to display past and ongoing projects and employment. Created pipelines to keep the status of current research up to date using AI to keep the website updated with the most recent pushes to GitHub. |
| 16 | 13 | FanThreeSixty Internship Summer 2026 | Developed backend functionality for a configuration mapping tool using Java, Quarkus, and PostgreSQL. The work involved processing configuration data, integrating functionality with existing services, testing application behavior, and debugging issues to ensure the backend operated correctly. |
| 17 | 5 | SoilProve Hackathon 2026 | Built a web application prototype using Next.js and Supabase to help farmers make corn nitrogen recommendations. The application incorporated agronomic data and MRTN-based guidance to estimate nitrogen requirements and compare potential fertilizer costs against conventional flat-rate application methods. |
| 18 | 5 | HackKU 2025 BloodIQ | Developed BloodIQ, a hackathon application using Python, Flask, and Google's Gemini AI model. The project involved building an AI-powered prototype, connecting web application functionality with a language model, and working under a limited development timeline to deliver a demonstrable application. |
| 19 | 1 | EECS 168 Lab 1 | Used basic python print statements to format personal information about the programmer. First experience with python programming and using github. |
| 20 | 1 | EECS 168 Lab 2 | Used python to create a mock search history simulation that reads an input file and outputs to a separate output file. Needed to learn file I/O to complete this lab. |
| 21 | 3 | EECS 665 L6 | Implemented a testing suite for our compiler project that was partially complete. Created a Makefile that used input, expected err, and expected out to diff for any issues. |
