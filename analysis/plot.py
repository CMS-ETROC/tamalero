import ROOT
import time
from sys import argv

def main():

    # Should add argument parser here

    # Set if you want to see the TCanvas or make pdf fast
    #setBatch = True
    setBatch = False
    ROOT.gROOT.SetBatch(setBatch)


    #Define strings to be used below
    meancal = 165.0
    toa_time_string = "12.5 - (3.125 / {})*toa_code".format(meancal)
    tot_time_string = "(3.125 / {})*(2*tot_code - TMath::Floor(tot_code / 32))".format(meancal)
    DT_string = toa_time_string + " - LP2_20[1]*1e9 + Clock"

    pixel_cut_string = "row == 6 && col == 7"
    #pixel_cut_string = tot_time_string+"> 5 && "+tot_time_string+" < 7"
    overall_cut_string = ""#pixel_cut_string

    full_cut_string = pixel_cut_string+ " && amp[1]>50 && amp[1]<300 && toa_code > 100 && toa_code < 500 && tot_code > 90 && cal_code >= 138 && cal_code <= 168"
    
    # Open all root files
    path = "/home/etl/Test_Stand/ETROC2_Test_Stand/ScopeHandler/ScopeData/LecroyMerged/"

    if(len(argv)==2):
        runNumList = [str(x) for x in range(int(argv[1]), int(argv[1]) + 1)] #lecroy
    elif(len(argv)>2):
        runNumList = [str(x) for x in range(int(argv[1]), int(argv[2]) + 1)] #lecroy
    else:
        runNumList = [str(x) for x in range(385, 397 + 1)] #lecroy
    print("Attempting to processes the following runs:", runNumList)
    files = [path+"run_"+runNum+".root" for runNum in runNumList]
    print(files)

    # Open root file and chain them together
    t = ROOT.TChain("pulse")
    for f in files:
        t.Add(f)

    # Make canvas
    c = ROOT.TCanvas("c","c",4*500,1000)
    c.Divide(4, 4)
        
    # Plot amp dist.
    c.cd(1); ROOT.gPad.SetLogy()
    t.Draw("amp[1]>>hamp1",overall_cut_string,"")

    c.cd(2); ROOT.gPad.SetLogy()
    t.Draw("amp[2]>>hamp2",overall_cut_string,"")

    # Plot time dist.
    c.cd(3)#; ROOT.gPad.SetLogy()
    t.Draw("LP2_20[1]*1e9>>htime1",overall_cut_string,"")

    c.cd(4)#; ROOT.gPad.SetLogy()
    t.Draw("Clock>>htime2",overall_cut_string,"")

    # Plot tot vs toa: all pixels
    i=5
    c.cd(i)
    t.Draw("tot_code:toa_code>>hbeam",overall_cut_string,"colz")
    
    # Plot beam position: col vs row
    i+=1
    c.cd(i)
    t.Draw("row:col>>hbeamX",overall_cut_string,"colz")
    
    # Plot beam position: col
    i+=1
    c.cd(i)
    t.Draw("col>>hbeamY",overall_cut_string)
    hbeamY = getattr(ROOT,"hbeamY")
    fitBeamY = ROOT.TF1("fitBeamY", "gaus")    
    fitBeamY.SetLineColor(ROOT.kRed)
    fitBeamY.Draw("same")    
    hbeamY.Fit(fitBeamY)
    fitBeamY.Draw("same")    

    # Plot beam position: row
    i+=1
    c.cd(i)
    t.Draw("row>>hrow",overall_cut_string)
    hrow = getattr(ROOT,"hrow")
    fitrow = ROOT.TF1("fitrow", "gaus")    
    fitrow.SetLineColor(ROOT.kRed)
    fitrow.Draw("same")    
    hrow.Fit(fitrow)
    fitrow.Draw("same")    

    # toa for a specific pixel
    i+=1
    c.cd(i)
    t.Draw("toa_code>>toa",pixel_cut_string,"hist")

    i+=1
    c.cd(i)
    t.Draw(toa_time_string+">>toaTime",pixel_cut_string,"hist")

    # tot for a specific pixel
    i+=1
    c.cd(i)
    t.Draw(tot_time_string+":row:col>>tot","","profcolz")
    #t.Draw("tot_code","","hist")
    i+=1
    c.cd(i)
    t.Draw(tot_time_string+">>totTime(100,0,10)",pixel_cut_string,"hist")

    # cal for a specific pixel
    i+=1
    c.cd(i)
    t.Draw("cal_code>>hist",pixel_cut_string,"hist")

    # Time Resolution for a specific pixel (No Time Walk Correction)
    i+=1
    c.cd(i)
    t.Draw(DT_string+">>timeRes(100,10,15)",full_cut_string)
    timeRes = getattr(ROOT,"timeRes")
    fitTimeRes = ROOT.TF1("fitTimeRes", "gaus")#,12,12.5)    
    fitTimeRes.SetLineColor(ROOT.kRed)
    fitTimeRes.Draw("same")    
    timeRes.Fit(fitTimeRes)
    fitTimeRes.Draw("same")
    fitTimeRes.Draw("same")

    # Time Walk Correction
    i+=1
    c.cd(i)
    #t.Draw(DT_string+"+11.3314 - (0.254533*"+tot_time_string+"):"+tot_time_string+">>TimeWalk(100,0,10, 100,9,13)",full_cut_string,"profcolz")
    t.Draw(DT_string+":"+tot_time_string+">>TimeWalk(100,0,10, 100,5,15)",full_cut_string,"profcolz")
    timeWalk = getattr(ROOT,"TimeWalk")
    low_bound = 3.5
    high_bound = 5
    linear_fit = ROOT.TF1("linear_fit", "[0] + [1]*x", low_bound, high_bound)
    linear_fit.SetParameters(10,0.5)
    fitTimeWalk = ROOT.TF1("fitTimeWalk", "linear_fit", low_bound, high_bound)
    fitTimeWalk.SetLineColor(ROOT.kRed)
    fitTimeWalk.Draw("same")
    timeWalk.Fit(fitTimeWalk, "R", "", low_bound, high_bound)
    intercept = fitTimeWalk.GetParameter(0)
    slope = fitTimeWalk.GetParameter(1)
    fitTimeWalk.Draw("same")
    fitTimeWalk.Draw("same")

    # Final Time Resolution
    i+=1
    c.cd(i)
    t.Draw(DT_string+"-{} - ({}*".format(intercept,slope)+tot_time_string+")>>timeResM(50,-0.5,0.5)",full_cut_string,"hist")
    timeResM = getattr(ROOT,"timeResM")
    fitTimeResM = ROOT.TF1("fitTimeResM", "gaus",-0.07,0.07)    
    fitTimeResM.SetLineColor(ROOT.kRed)
    fitTimeResM.Draw("same")    
    timeResM.Fit(fitTimeResM)
    fitTimeResM.Draw("same")
    fitTimeResM.Draw("same")
    
    # Save canvas as a pdf
    c.Print("dqm.pdf")

    # Keep code open forever if you want to see the TCanvas
    print("Finished making all histograms")
    if not setBatch:
        if(len(runNumList)==1):
            print("Run #: {}".format(runNumList[0]))
        else:
            print("Run # range: [{} : {}]".format(runNumList[0],runNumList[-1]))
        print(time.asctime())
        print("Safe to close")
        if(len(argv)<4):
            time.sleep(10**9)
        else:
            time.sleep(int(argv[3]))

if __name__ == '__main__':
    main()

