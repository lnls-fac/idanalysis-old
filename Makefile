DISTPATH=$(shell python -c "import site; print(site.getsitepackages())" | cut -f2 -d"'")
PACKAGE:=$(shell basename $(shell pwd))
ISINST=$(shell pip show $(PACKAGE) | wc -l )
EGGLINK=$(DISTPATH)/$(PACKAGE).egg-link
TMPFOLDER=/tmp/install-$(PACKAGE)

clean:
	git clean -fdX

develop: uninstall
	pip install --no-deps -e ./

install: uninstall
ifneq (, $(wildcard $(TMPFOLDER)))
	rm -rf /tmp/install-$(PACKAGE)
endif
	cp -rRL ../$(PACKAGE) /tmp/install-$(PACKAGE)
	cd /tmp/install-$(PACKAGE)/; pip install --no-deps ./
	rm -rf /tmp/install-$(PACKAGE)

# known issue: It will fail to uninstall scripts
#  if they were installed in develop mode
uninstall: clean
ifneq (,$(wildcard $(EGGLINK)))
	rm -r $(EGGLINK)
endif
ifneq ($(ISINST),0)
	pip uninstall -y $(PACKAGE)
	sed -i '/$(PACKAGE)/d' $(DISTPATH)/easy-install.pth
else
	echo 'already uninstalled $(PACKAGE)'
endif
