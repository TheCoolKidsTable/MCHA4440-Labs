TARGET_NAME := lab3

# OS specific config
ifeq ($(OS),Windows_NT)
	TARGET_SUFFIX := .exe
else
	TARGET_SUFFIX := .out
endif

# compiler
CXX := g++

# defines
DEFINES :=

# packages used (as recognised by pkg-config)
PACKAGES := opencv4

# custom include paths
INCLUDE :=

# package include paths
ifneq ($(strip $(PACKAGES)),)
INCLUDE += $(shell pkg-config --cflags-only-I $(PACKAGES))
endif

# includes for unit testing framework
TEST_INCLUDE := $(shell pkg-config --cflags-only-I catch2)

# compiler flags
CXXFLAGS := -std=c++17 -O3 -march=native -mtune=native

# linker flags
LDFLAGS :=

# custom linked libraries
LINKLIBS :=

# package linked libraries
ifneq ($(strip $(PACKAGES)),)
LINKLIBS += $(shell pkg-config --libs $(PACKAGES))
endif

# path macros
BIN_DIR := bin
OBJ_DIR := build
SRC_DIR := src

TEST_DIR := test
TEST_BIN_DIR := $(TEST_DIR)/bin
TEST_OBJ_DIR := $(TEST_DIR)/build
TEST_SRC_DIR := $(TEST_DIR)/src

TARGET := $(BIN_DIR)/$(addsuffix $(TARGET_SUFFIX),$(TARGET_NAME))
TEST_TARGET := $(TEST_BIN_DIR)/$(addsuffix $(TARGET_SUFFIX),test)

# src files & obj files
SRC := $(foreach x, $(SRC_DIR), $(wildcard $(addprefix $(x)/*,.c*)))
OBJ := $(addprefix $(OBJ_DIR)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))
DEP := $(OBJ:.o=.d)  # one dependency file for each source

TEST_SRC := $(foreach x, $(TEST_SRC_DIR), $(wildcard $(addprefix $(x)/*,.c*)))
TEST_OBJ := $(addprefix $(TEST_OBJ_DIR)/, $(addsuffix .o, $(notdir $(basename $(TEST_SRC)))))
TEST_DEP := $(TEST_OBJ:.o=.d)  # one dependency file for each source

# clean files list
DISTCLEAN_LIST := $(OBJ) \
				  $(DEP) \
				  $(TEST_OBJ) \
				  $(TEST_DEP)

CLEAN_LIST := $(TARGET) \
			  $(TEST_TARGET) \
			  $(DISTCLEAN_LIST)

# default rule
default: all

# non-phony targets
$(TARGET): $(OBJ) | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LINKLIBS) -o $@ 

$(TEST_TARGET): $(TEST_OBJ) $(filter-out $(OBJ_DIR)/main.o, $(OBJ)) | $(TEST_BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LINKLIBS) -o $@ 

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c* | $(OBJ_DIR)
	$(CXX) -c -MMD $< -o $@ $(CXXFLAGS) $(INCLUDE)

$(TEST_OBJ_DIR)/%.o: $(TEST_SRC_DIR)/%.c* | $(TEST_OBJ_DIR)
	$(CXX) -c -MMD $< -o $@ $(CXXFLAGS) $(INCLUDE) $(TEST_INCLUDE)

# include all dep files in the makefile
-include $(DEP)
-include $(TEST_DEP)

# create directories if they don't exist
$(OBJ_DIR) $(BIN_DIR) $(TEST_OBJ_DIR) $(TEST_BIN_DIR):
	@mkdir "$@"

# phony rules
.PHONY: all
all: $(TARGET) test

.PHONY: test
test: $(TEST_TARGET)
	$(TEST_TARGET) --use-colour yes

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)

.PHONY: distclean
distclean:
	@echo CLEAN $(DISTCLEAN_LIST)
	@rm -f $(DISTCLEAN_LIST)