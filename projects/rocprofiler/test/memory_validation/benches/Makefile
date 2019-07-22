# specify each subdir here
SUBDIRS = test_cache

TARGETS = all clean

.PHONY: $(TARGETS) default

# change this line to change the default action
# (using one of the targets above)
default: all

define do-target
$(1): $$(foreach subdir, $$(SUBDIRS), $(1)-$$(subdir))

$(1)-%:
	$$(MAKE) -C $$* $(1)
endef

$(foreach tgt, $(TARGETS), $(eval $(call do-target,$(tgt))))
